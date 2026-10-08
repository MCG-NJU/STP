"""
STP pre-training entry point.

Single node, 8 GPUs (effective batch = 64 * 8 = 512; pass --accum_iter to reach
the paper's 4096):

    torchrun --nproc_per_node=8 main_pretrain_stp.py \
        --manifest /path/to/egoclip.csv \
        --data_root /path/to/ego4d_256_egoclip_AVG3s_img \
        --output_dir /path/to/exp/stp_vitb \
        --batch_size 64 --accum_iter 8 --epochs 50 --warmup_epochs 5

Release recipe: AdamW, blr 1.5e-4, wd 0.05, betas (0.9, 0.95), cosine decay,
50 epochs, mask 75% / 95%, frame interval 16, RandomResizedCrop(0.8, 1.0).
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__, 'timm is required (the reference code pins 0.3.2)'

import timm.optim.optim_factory as optim_factory

import models_stp
import util.misc as misc
from datasets_egoclip import DEFAULT_FRAME_INTERVAL, EgoClipFrames
from engine_pretrain_stp import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    p = argparse.ArgumentParser('STP pre-training', add_help=False)

    p.add_argument('--batch_size', default=64, type=int,
                   help='batch size per GPU (effective = batch_size * accum_iter * world_size)')
    p.add_argument('--epochs', default=50, type=int)
    p.add_argument('--accum_iter', default=1, type=int)

    # model
    p.add_argument('--model', default='mae_vit_base_patch16', type=str)
    p.add_argument('--input_size', default=224, type=int)
    p.add_argument('--mask_ratio_current', default=0.75, type=float,
                   help='masking ratio for the current frame (spatial branch)')
    p.add_argument('--mask_ratio_future', default=0.95, type=float,
                   help='masking ratio for the future frame (temporal branch)')
    p.add_argument('--norm_pix_loss', action='store_true',
                   help='use per-patch normalized pixels as reconstruction targets')
    p.set_defaults(norm_pix_loss=True)

    # optimizer
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=None,
                   help='absolute lr; if unset, lr = blr * effective_batch_size / 256')
    p.add_argument('--blr', type=float, default=1.5e-4)
    p.add_argument('--min_lr', type=float, default=0.)
    p.add_argument('--warmup_epochs', type=int, default=5)

    # data
    p.add_argument('--manifest', default='', type=str,
                   help='EgoClip CSV (video_uid, narration_source, narration_ind, clip_text, length)')
    p.add_argument('--data_root', default='', type=str,
                   help='root of the frame tree or the clip mp4 tree')
    p.add_argument('--backend', default='frame_dir', choices=['frame_dir', 'video'])
    p.add_argument('--frame_interval', default=DEFAULT_FRAME_INTERVAL, type=int,
                   help='frames between the current and future frame')
    p.add_argument('--alpha', default=0.3, type=float,
                   help='sampling span factor for short clips')
    p.add_argument('--aug_scale_min', default=0.8, type=float,
                   help='RandomResizedCrop lower bound; the release recipe uses 0.8')

    # bookkeeping
    p.add_argument('--output_dir', default='', help='checkpoint + log destination')
    p.add_argument('--log_dir', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--resume', default='')
    p.add_argument('--start_epoch', default=0, type=int)
    p.add_argument('--num_workers', default=8, type=int)
    p.add_argument('--save_every', default=2, type=int,
                   help='save a checkpoint every N epochs (and always on the last)')
    p.add_argument('--pin_mem', action='store_true')
    p.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    p.set_defaults(pin_mem=True)

    # distributed
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--local_rank', default=-1, type=int)
    p.add_argument('--dist_on_itp', action='store_true')
    p.add_argument('--dist_url', default='env://')

    return p


def build_transform(scale_min):
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(scale_min, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = EgoClipFrames(
        manifest_path=args.manifest,
        root=args.data_root,
        transform=build_transform(args.aug_scale_min),
        backend=args.backend,
        frame_interval=args.frame_interval,
        alpha=args.alpha,
    )
    print('dataset: {} clips'.format(len(dataset_train)))

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    log_writer = None
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True)

    model = models_stp.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        mask_ratio_current=args.mask_ratio_current,
        mask_ratio_future=args.mask_ratio_future)
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print('base lr: %.2e' % (args.lr * 256 / eff_batch_size))
    print('actual lr: %.2e' % args.lr)
    print('effective batch size: %d' % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader_train, optimizer, device,
                                      epoch, loss_scaler, log_writer=log_writer, args=args)

        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

    print('Training time {}'.format(datetime.timedelta(seconds=int(time.time() - start_time))))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
