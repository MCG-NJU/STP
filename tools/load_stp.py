"""
STP checkpoint loader + forward-usage example.

    # 1) verify the encoder loads from the released checkpoint (needs timm)
    python tools/load_stp.py --ckpt 0.95_aug0.8.pth --check-only

    # 2) run one pre-training forward on a real frame pair
    python tools/load_stp.py --ckpt 0.95_aug0.8.pth \
        --current samples/current_frame.png --future samples/future_frame.png

    # 3) extract frozen [CLS] features for a downstream policy
    python tools/load_stp.py --ckpt 0.95_aug0.8.pth \
        --current samples/current_frame.png --features

The released checkpoint is a full pre-training state dict:
    {model, optimizer, epoch, scaler, args}
Only `model` is used here; both decoders are dropped for downstream use.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models_stp  # noqa: E402
from util.ckpt_io import load_checkpoint  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(args):
    model = models_stp.__dict__[args.model](
        norm_pix_loss=True,
        mask_ratio_current=args.mask_ratio_current,
        mask_ratio_future=args.mask_ratio_future,
    )
    return model


def load_checkpoint(model, ckpt_path, verbose=True):
    """Load the released STP pre-training checkpoint into the model.

    Returns the raw checkpoint dict (useful for reading `args` / `epoch`).
    Uses strict=True: the released weights match this architecture exactly
    (434 tensors). If you changed the architecture, expect a shape error.
    """
    ckpt = load_checkpoint(ckpt_path)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    # strict=True: the released weights match this architecture exactly
    # (434 tensors). If you changed the architecture, expect a shape error.
    model.load_state_dict(state, strict=True)

    if verbose:
        n = len(state)
        print(f'[stp] loaded {n} tensors from {ckpt_path}')
        if isinstance(ckpt, dict):
            meta = {k: v for k, v in ckpt.items() if k != 'model'}
            if 'epoch' in meta:
                print(f'[stp] checkpoint epoch = {meta["epoch"]}')
            if isinstance(meta.get('args'), dict):
                a = meta['args']
                keys = ('model', 'input_size', 'mask_ratio', 'norm_pix_loss',
                        'blr', 'weight_decay', 'epochs', 'warmup_epochs', 'world_size')
                shown = {k: a[k] for k in keys if k in a}
                print(f'[stp] pre-training args = {shown}')
    return ckpt


def preprocess(path, size=224):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def main():
    p = argparse.ArgumentParser('STP checkpoint loader / forward usage')
    p.add_argument('--ckpt', required=True, help='path to 0.95_aug0.8.pth')
    p.add_argument('--model', default='mae_vit_base_patch16')
    p.add_argument('--current', default='samples/current_frame.png',
                   help='current frame I_c at time t')
    p.add_argument('--future', default='samples/future_frame.png',
                   help='future frame I_f at time t+16')
    p.add_argument('--mask_ratio_current', type=float, default=0.75)
    p.add_argument('--mask_ratio_future', type=float, default=0.95)
    p.add_argument('--check-only', action='store_true',
                   help='only verify that the checkpoint matches the architecture')
    p.add_argument('--features', action='store_true',
                   help='run forward_features() and inspect the [CLS] representation')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    model = build_model(args)
    own = model.state_dict()
    n_all = sum(v.numel() for v in own.values())
    n_enc = sum(v.numel() for k, v in own.items()
                if k.startswith(('patch_embed', 'blocks', 'cls_token'))
                or k in ('norm.weight', 'norm.bias'))
    print(f'[stp] model={args.model}  params={n_all/1e6:.2f}M  '
          f'(encoder {n_enc/1e6:.2f}M, decoders {(n_all-n_enc)/1e6:.2f}M)')

    load_checkpoint(model, args.ckpt)
    model.eval()

    if args.check_only:
        print('[stp] checkpoint / architecture match OK')
        return

    img_c = preprocess(args.current)
    img_f = preprocess(args.future)

    if args.features:
        feat = model.forward_features(img_c.unsqueeze(0))
        print(f'[stp] forward_features -> [CLS] shape={tuple(feat.shape)}  '
              f'norm={feat.norm().item():.4f}  '
              f'min={feat.min().item():.4f}  max={feat.max().item():.4f}')
        return

    with torch.no_grad():
        loss, pred_c, pred_f, mask_c = model(img_c.unsqueeze(0), img_f.unsqueeze(0))

    print(f'[stp] forward -> loss={loss.item():.4f}')
    print(f'[stp]   pred_spatial  {tuple(pred_c.shape)}   '
          f'masked patches {int(mask_c.sum().item())}/196')
    print(f'[stp]   pred_temporal {tuple(pred_f.shape)}')
    print('[stp] reconstructed current frame at samples/recon_current.png')

    rec = model.unpatchify(pred_c)[0].permute(1, 2, 0).cpu().numpy()
    # de-normalize patch-wise (norm_pix_loss training: prediction is in
    # per-patch normalized space, so this is a rough visualisation only)
    rec = (rec - rec.min()) / (rec.max() - rec.min() + 1e-6)
    Image.fromarray((rec * 255).astype(np.uint8)).save('samples/recon_current.png')


if __name__ == '__main__':
    main()
