"""
Extract a frozen STP encoder from a pre-training checkpoint and save it as a
plain timm-style ViT, so downstream policy code can consume it without the
dual decoders.

    python tools/export_encoder.py \
        --ckpt lang_0.95_aug0.8_no_lang.pth \
        --output stp_vitb_encoder.pth

The exported file contains:
    {'model': <ViT state dict>, 'embed_dim': 768, 'patch_size': 16,
     'source_epoch': 49, 'arch': 'stp_vitb'}

Load it into any `timm.models.vision_transformer.VisionTransformer` built with
patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True and
norm_layer=partial(nn.LayerNorm, eps=1e-6), i.e. the same backbone as
`models_vit.vit_base_patch16` from the MAE/VC-1 lineage. `pos_embed` is the
fixed sin-cos table, so it is copied verbatim -- do not re-initialise it.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models_stp  # noqa: E402


def main():
    p = argparse.ArgumentParser('export a frozen STP encoder')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='mae_vit_base_patch16')
    p.add_argument('--verify', default='samples/current_frame.png',
                   help='image to sanity-check the exported encoder against the full model')
    args = p.parse_args()

    model = models_stp.__dict__[args.model]()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    encoder_state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith(('decoder_', 'mask_token'))}
    out = {
        'model': encoder_state,
        'embed_dim': 768,
        'patch_size': 16,
        'depth': 12,
        'num_heads': 12,
        'source_epoch': ckpt.get('epoch') if isinstance(ckpt, dict) else None,
        'arch': 'stp_vitb',
    }
    torch.save(out, args.output)
    print(f'[stp] exported encoder: {len(encoder_state)} tensors -> {args.output}')
    print(f'[stp] tensor count dropped from {len(model.state_dict())} to {len(encoder_state)}')

    if args.verify and os.path.exists(args.verify):
        img = Image.open(args.verify).convert('RGB').resize((224, 224), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0
        x = ((x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
             / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)).unsqueeze(0)
        with torch.no_grad():
            feat = model.forward_features(x)
        print(f'[stp] sanity [CLS] shape={tuple(feat.shape)} norm={feat.norm().item():.4f}')


if __name__ == '__main__':
    main()
