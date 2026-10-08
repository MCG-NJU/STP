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

Pre-trained visual encoder for robotic manipulation, from the paper
**Spatiotemporal Predictive Pre-training for Robotic Motor Control**
(IJCV 2026, [arXiv:2403.05304](https://arxiv.org/abs/2403.05304)).

Official code: [MCG-NJU/STP](https://github.com/MCG-NJU/STP)

STP pre-trains a **plain image ViT** on large-scale egocentric video (Ego4D /
EgoClip) with **no** language, action, depth or teacher supervision. It jointly
learns content and motion features by predicting the masked current frame and
the masked future frame with a shared encoder and **decoupled dual decoders**.

At evaluation the encoder is **frozen**, both decoders are discarded, and the
`[CLS]` token is used as the visual representation for downstream policy
learning.

## Files

| file | size | contents |
|---|---|---|
| `stp_vitb.pth` | ~558 MiB | model weights (encoder + both decoders), 146.34 M params |
| `stp_vitb_encoder.pth` | ~343 MiB | decoder-free encoder, for consumers using a plain timm ViT |
| `config.json` | small | architecture and pre-training recipe |
| `models_stp.py`, `util/` | small | model definition, so this repo is self-contained |
| `loader.py` | small | minimal `STPEncoder` wrapper |

The full 1.75 GB training checkpoint (with AdamW optimizer state, for resuming
pre-training) is available separately — see the repository.

## Architecture

| | |
|---|---|
| encoder | ViT-B/16, 12 layers, 12 heads, 768 dim, fixed sincos pos embed |
| spatial decoder | 8 blocks, 16 heads, 512 dim, transformer encoder blocks |
| temporal decoder | 8 blocks, 16 heads, 512 dim, self-attn + cross-attn + FFN |
| input | 224 × 224, ImageNet normalisation |

The **spatial** branch masks the current frame at 75% and reconstructs it.
The **temporal** branch masks the future frame at 95% and reconstructs it,
conditioned on the current-frame feature `Z_c` used as a **fixed** key/value —
so the future-frame query cannot write back into the current-frame
representation. That asymmetry is the design that matters most: the ablation
scores 63.7 for `self-cross` against 59.8 for `joint-self`.

## Usage

```python
import torch
from models_stp import mae_vit_base_patch16
from huggingface_hub import hf_hub_download

path = hf_hub_download('MCG-NJU/STP', 'stp_vitb.pth')
model = mae_vit_base_patch16()
model.load_state_dict(torch.load(path, map_location='cpu')['model'], strict=True)
model.eval()

# images: (N, 3, 224, 224), ImageNet-normalised -> (N, 768)
feats = model.forward_features(images)
```

Or with the bundled wrapper:

```python
from loader import STPEncoder

enc = STPEncoder.from_pretrained('MCG-NJU/STP')
feats = enc.encode([pil_image_1, pil_image_2])   # (N, 768)
```

Match the convention used by VC-1 and R3M: concatenate multi-view and
multi-frame features in the channel dimension together with the robot's
proprioceptive state, then train the policy. STP does not constrain the policy
architecture — the paper evaluates MLPs, transformers, RVT-2 and ACT.

```python
# equivalent, without the hub package
import torch, models_stp
model = models_stp.mae_vit_base_patch16()
model.load_state_dict(torch.load('stp_vitb.pth', map_location='cpu')['model'], strict=True)
```

## Pre-training recipe

AdamW, base lr 1.5e-4, weight decay 0.05, betas (0.9, 0.95), cosine decay,
50 epochs with 5 warmup, effective batch 4096, `RandomResizedCrop(0.8, 1.0)`
plus horizontal flip, frame interval 16, mask ratios 75% / 95%, per-patch
normalised reconstruction target, MSE summed over both branches at 1:1 weight.

Pre-training costs about 1.77× standard image MAE (34.53 vs 19.56 GFLOPs for
ViT-B/16 at 224²), thanks to the shallow temporal decoder and the 95% future
masking.

## Results

Weighted average over five single-task simulation benchmarks (Meta-World,
Franka-Kitchen, DMControl, Adroit, Trifinger), same pre-training data
throughout:

| method | WA |
|---|---|
| **STP (frozen, ViT-B/16)** | **63.7** |
| MAE baseline | 59.6 |
| VC-1 | 61.8 |
| DINOv2 | 59.6 |
| CLIP | 55.6 |
| R3M | 54.9 |

Language-conditioned multi-task: LIBERO-LONG 41.3 ± 0.3, RLBench 51.3 ± 1.4.
Real-world (five tasks on a low-cost dual-arm robot): 49.3 in the standard
environment, 46.3 with unseen distractors, against VC-1's 42.0 / 36.7.

## Note on language

This checkpoint is the **single-modality** variant. Despite the `lang_` in the
original research filename, it contains no language parameters. Adding language
narration as a temporal-prediction condition *hurt* performance in the ablation
(63.1 with language at the encoder vs 63.7 without), because of the modality gap
between pre-training and single-modal downstream policies.

## Intended use

Frozen visual representation for behaviour-cloning policy learning in robotic
manipulation, especially few-shot regimes. The encoder is not fine-tuned in the
paper's main results; post-training the encoder on task data is a documented
alternative that performs better than end-to-end fine-tuning.

Not intended for general-purpose vision tasks; it is an image-level
representation optimised for manipulation.

## Citation

```bibtex
@Article{Yang2026STP,
  author  = {Jiange Yang and Bei Liu and Jianlong Fu and Bocheng Pan
             and Gangshan Wu and Limin Wang},
  journal = {International Journal of Computer Vision},
  title   = {Spatiotemporal Predictive Pre-training for Robotic Motor Control},
  year    = {2026},
  volume  = {134},
  number  = {354},
  doi     = {10.1007/s11263-026-02948-3},
}
```

## License

Apache-2.0.
