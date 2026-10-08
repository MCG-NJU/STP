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

Concatenate multi-view and multi-frame features in the channel dimension
together with the robot's proprioceptive state, then train the policy. STP does
not constrain the policy architecture.

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
