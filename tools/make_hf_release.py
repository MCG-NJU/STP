"""
Split a released STP pre-training checkpoint into inference-friendly files.

`lang_0.95_aug0.8_no_lang.pth` is a *training* checkpoint: 1.75 GB, of which
only 558 MiB is the model. The rest is AdamW optimizer state (`exp_avg` and
`exp_avg_sq` for 432 of the 434 tensors) plus the AMP scaler, which is dead
weight for anyone who only wants to run the encoder.

    python tools/make_hf_release.py \
        --ckpt weights/lang_0.95_aug0.8_no_lang.pth \
        --output-dir hf_release

Writes:
    stp_vitb.pth              model weights only  (~558 MiB)
    stp_vitb_encoder.pth      decoder-free ViT    (~343 MiB)
    checkpoint_spec.json      architecture + recipe
    <model card files>        copied from hf/ in the repo root

The full training checkpoint stays available separately if you want to resume
pre-training; this is purely about what a consumer needs.
"""
import argparse
import json
import os
import shutil
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models_stp  # noqa: E402

# keys that are fixed buffers and never receive optimizer state
FROZEN_KEYS = ('pos_embed', 'decoder_pos_embed')


def human(nbytes):
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if nbytes < 1024:
            return f'{nbytes:.1f} {unit}'
        nbytes /= 1024
    return f'{nbytes:.1f} TiB'


def tensor_bytes(sd):
    return sum(v.numel() * v.element_size() for v in sd.values())


def main():
    p = argparse.ArgumentParser('build the HuggingFace release files')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--output-dir', default='hf_release')
    p.add_argument('--model', default='mae_vit_base_patch16')
    p.add_argument('--model-card-dir', default='hf',
                   help='directory holding the model card files to copy over')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt['model']
    epoch = ckpt.get('epoch')
    train_args = ckpt.get('args')
    if hasattr(train_args, '__dict__'):
        train_args = vars(train_args)

    print(f'[hf] source            {args.ckpt}')
    print(f'[hf] model tensors     {len(state)}  ({human(tensor_bytes(state))})')
    if 'optimizer' in ckpt:
        opt = ckpt['optimizer']['state']
        n = sum(1 for k in opt)
        print(f'[hf] optimizer entries {n}  (dropping)')

    # 1) model weights only -- this is what a consumer loads through
    #    models_stp + load_state_dict, same as the original file
    model_path = os.path.join(args.output_dir, 'stp_vitb.pth')
    torch.save({'model': state, 'epoch': epoch, 'args': train_args,
                'note': 'STP ViT-B/16 pre-trained weights (optimizer state stripped)'},
               model_path)
    print(f'[hf] wrote {model_path}  ({human(os.path.getsize(model_path))})')

    # 2) decoder-free encoder, for consumers using a plain timm ViT
    model = models_stp.__dict__[args.model]()
    model.load_state_dict(state, strict=True)
    enc_state = {k: v for k, v in model.state_dict().items()
                 if not k.startswith(('decoder_', 'mask_token'))}
    enc_path = os.path.join(args.output_dir, 'stp_vitb_encoder.pth')
    torch.save({'model': enc_state, 'embed_dim': 768, 'patch_size': 16,
                'depth': 12, 'num_heads': 12, 'source_epoch': epoch,
                'arch': 'stp_vitb'}, enc_path)
    print(f'[hf] wrote {enc_path}  ({human(os.path.getsize(enc_path))}, '
          f'{len(enc_state)}/{len(state)} tensors)')

    # 3) config, so the numbers live next to the weights
    config = {
        'architectures': ['STP'],
        'model_type': 'stp',
        'image_size': 224,
        'patch_size': 16,
        'encoder': {'embed_dim': 768, 'depth': 12, 'num_heads': 12,
                    'pos_embed': 'fixed sincos'},
        'spatial_decoder': {'embed_dim': 512, 'depth': 8, 'num_heads': 16},
        'temporal_decoder': {'embed_dim': 512, 'depth': 8, 'num_heads': 16,
                             'type': 'self-attn + cross-attn(context=Z_c) + FFN'},
        'mask_ratio_current': 0.75,
        'mask_ratio_future': 0.95,
        'frame_interval': 16,
        'num_parameters': sum(v.numel() for v in state.values()),
        'source_epoch': epoch,
        'pretraining': {'optimizer': 'AdamW', 'blr': 1.5e-4, 'weight_decay': 0.05,
                        'betas': [0.9, 0.95], 'epochs': 50, 'warmup_epochs': 5,
                        'effective_batch_size': 4096,
                        'augmentation': 'RandomResizedCrop(0.8, 1.0)',
                        'dataset': 'EgoClip (Ego4D)',
                        'normalize_mean': [0.485, 0.456, 0.406],
                        'normalize_std': [0.229, 0.224, 0.225]},
        'usage': 'freeze the encoder, take the [CLS] token via forward_features()',
        'paper': 'arXiv:2403.05304',
    }
    cfg_path = os.path.join(args.output_dir, 'config.json')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f'[hf] wrote {cfg_path}')

    # 4) model card + loader, authored under hf/
    if os.path.isdir(args.model_card_dir):
        for name in sorted(os.listdir(args.model_card_dir)):
            src = os.path.join(args.model_card_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.output_dir, name))
                print(f'[hf] copied {name}')

    # 5) the model definition, so the hub copy is self-contained and
    #    hf/loader.py can import it without cloning the repo
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ('models_stp.py',):
        shutil.copy2(os.path.join(repo_root, name), os.path.join(args.output_dir, name))
        print(f'[hf] copied {name}')
    shutil.copytree(os.path.join(repo_root, 'util'),
                    os.path.join(args.output_dir, 'util'),
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    print('[hf] copied util/')

    print()
    print('[hf] totals:')
    for name in sorted(os.listdir(args.output_dir)):
        path = os.path.join(args.output_dir, name)
        if os.path.isfile(path):
            print(f'       {human(os.path.getsize(path)):>10}  {name}')


if __name__ == '__main__':
    main()
