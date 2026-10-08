---
license: apache-2.0
library_name: pytorch
tags:
  - robotics
  - robot-manipulation
  - self-supervised
  - masked-autoencoder
  - egocentric-video
  - visual-representation
  - pytorch
pipeline_tag: feature-extraction
---

# STP: Spatiotemporal Predictive Pre-training for Robotic Motor Control

Pre-trained visual encoder for robotic manipulation, from the paper published in
*International Journal of Computer Vision* (2026):
[10.1007/s11263-026-02948-3](https://link.springer.com/article/10.1007/s11263-026-02948-3)

Official code: [MCG-NJU/STP](https://github.com/MCG-NJU/STP)

STP pre-trains a plain image ViT on large-scale egocentric video (Ego4D /
EgoClip), with no language, action, depth or teacher supervision. It jointly
learns content and motion features by reconstructing the masked current frame
and the masked future frame, using a shared encoder with decoupled dual
decoders.

At evaluation the encoder is frozen, both decoders are discarded, and the
`[CLS]` token is used as the visual representation for downstream policy
learning.

## Files

| file | size | contents |
|---|---|---|
| `stp_vitb.pth` | ~558 MiB | model weights (encoder + both decoders), 146.34 M parameters |
| `stp_vitb_encoder.pth` | ~327 MiB | decoder-free encoder, for use with a plain timm ViT |
| `config.json` | small | architecture and pre-training recipe |
| `models_stp.py`, `util/` | small | model definition, so this repo is self-contained |
| `loader.py` | small | minimal `STPEncoder` wrapper |

## Architecture

| | |
|---|---|
| encoder | ViT-B/16, 12 layers, 12 heads, 768 dim, fixed sincos positional embedding |
| spatial decoder | 8 blocks, 16 heads, 512 dim, transformer encoder blocks |
| temporal decoder | 8 blocks, 16 heads, 512 dim, self-attention + cross-attention + FFN |
| input | 224 × 224, ImageNet normalisation |

The spatial branch masks the current frame at 75% and reconstructs it. The
temporal branch masks the future frame at 95% and reconstructs it, conditioned
on the current-frame feature `Z_c` used as a fixed key/value, so the future-frame
query cannot write back into the current-frame representation.

## Usage

```python
import torch
from models_stp import mae_vit_base_patch16
from huggingface_hub import hf_hub_download

path = hf_hub_download('yangjiange/STP', 'stp_vitb.pth')
model = mae_vit_base_patch16()
model.load_state_dict(torch.load(path, map_location='cpu')['model'], strict=True)
model.eval()

# images: (N, 3, 224, 224), ImageNet-normalised -> (N, 768)
feats = model.forward_features(images)
```

Or with the bundled wrapper:

```python
from loader import STPEncoder

enc = STPEncoder.from_pretrained('yangjiange/STP')
feats = enc.encode([pil_image_1, pil_image_2]) # (N, 768)
```

Concatenate multi-view and multi-frame features in the channel dimension
together with the robot's proprioceptive state, then train the policy. STP does
not constrain the policy architecture.

## Pre-training recipe

AdamW, base learning rate 1.5e-4, weight decay 0.05, betas (0.9, 0.95), cosine
decay, 50 epochs with 5 warmup epochs, effective batch size 4096,
`RandomResizedCrop(0.8, 1.0)` with horizontal flip, frame interval 16, mask
ratios 75% / 95%, per-patch normalised reconstruction target, MSE summed over
both branches with 1:1 weight.

Pre-training costs about 1.77× standard image MAE (34.53 vs 19.56 GFLOPs for
ViT-B/16 at 224²), owing to the shallow temporal decoder and the 95% future
masking.

## Note on language

This checkpoint is the single-modality variant and contains no language
parameters. Adding language narration as a temporal-prediction condition
reduced performance in the paper's ablation, because of the modality gap
between pre-training and single-modal downstream policies.

## Intended use

Frozen visual representation for behaviour-cloning policy learning in robotic
manipulation, particularly in few-shot regimes. The encoder is not fine-tuned in
the paper's main results; post-training the encoder on task data is a documented
alternative that performs better than end-to-end fine-tuning.

It is not intended for general-purpose vision tasks.

## Verification

Checked on `torch 2.14.1+cpu` with `timm 0.3.2`:

- `load_state_dict(..., strict=True)` succeeds: 434/434 tensors, no missing and
  no unexpected keys.
- A pre-training forward on a real frame pair gives a finite loss, with masks of
  exactly 147/196 (75%) and ~186/196 (95%).
- `stp_vitb.pth` is bit-identical to the original training checkpoint in model
  weights; only the optimizer state and AMP scaler were removed.
- `stp_vitb_encoder.pth` loads into a plain `timm` `VisionTransformer`
  (`patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True`), with
  only the classification head unmatched, as expected for a frozen encoder.

Two version notes if you load these yourself:

- On PyTorch >= 2.6, `torch.load` defaults to `weights_only=True`, and the
  checkpoint's `args` field is an `argparse.Namespace`, which is not on the
  allowlist. Allowlist it, or use the bundled `util/ckpt_io.py`.
- `timm==0.3.2` imports `container_abcs` from `torch._six`, removed in PyTorch
  2.0. `util/torch_six_compat.py` restores that one attribute; import it before
  `timm`.

## Citation

```bibtex
@Article{Yang2026STP,
  author  = {Jiange Yang and Bei Liu and Jianlong Fu and Bocheng Pan
             and Gangshan Wu and Limin Wang},
  journal = {International Journal of Computer Vision},
  title   = {Spatiotemporal Predictive Pre-training for Robotic Motor Control},
  year    = {2026},
  volume  = {134},
  doi     = {10.1007/s11263-026-02948-3},
}
```

## License

Apache-2.0.
