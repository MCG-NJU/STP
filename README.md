# STP: Spatiotemporal Predictive Pre-training for Robotic Motor Control

Official pre-training code and weights.

> Jiange Yang, Bei Liu, Jianlong Fu, Bocheng Pan, Gangshan Wu, Limin Wang.
> *International Journal of Computer Vision* (2026).
> [10.1007/s11263-026-02948-3](https://link.springer.com/article/10.1007/s11263-026-02948-3)

STP pre-trains a plain image ViT on large-scale egocentric video (Ego4D /
EgoClip), with no language, action, depth or teacher supervision. The spatial
branch masks the current frame at 75% and reconstructs it; the temporal branch
masks the future frame at 95% and reconstructs it, conditioned on the
current-frame feature used as a fixed key/value. Downstream, the encoder is
frozen, both decoders are discarded, and the `[CLS]` token is the visual
representation.

## Weights

Hosted on HuggingFace: [yangjiange/STP](https://huggingface.co/yangjiange/STP)

| file | size | contents |
|---|---|---|
| `stp_vitb.pth` | ~558 MiB | ViT-B/16, encoder + both decoders, 146.34 M parameters |
| `stp_vitb_encoder.pth` | ~327 MiB | decoder-free encoder for a plain timm ViT |

## Install

```
pip install torch torchvision timm==0.3.2 numpy pillow
```

`timm==0.3.2` is required; its PyTorch 2.x incompatibility is handled by a
bundled shim.

## Usage

```python
import torch
from models_stp import mae_vit_base_patch16
from huggingface_hub import hf_hub_download

path = hf_hub_download('yangjiange/STP', 'stp_vitb.pth')
model = mae_vit_base_patch16()
model.load_state_dict(torch.load(path, map_location='cpu')['model'], strict=True)
model.eval()

feats = model.forward_features(images)   # (N, 3, 224, 224) -> (N, 768)
```

Command line:

```
# images / video -> features
python tools/inference.py --ckpt yangjiange/STP --input 'frames/*.png' --output feats.npy

# verify a checkpoint against the architecture (no torch needed)
python tools/verify_ckpt_params.py 0.95_aug0.8.pth
```

## Pre-train

Extract EgoClip frames once, then:

```
python tools/prepare_ego4d_frames.py --manifest egoclip.csv \
    --clip_root <clips> --output_root <frames>

torchrun --nproc_per_node=8 main_pretrain_stp.py \
    --manifest egoclip.csv --data_root <frames> --output_dir <exp> \
    --batch_size 64 --accum_iter 8 --epochs 50 --warmup_epochs 5
```

Effective batch size 4096, matching the paper.

## Docs

- [docs/FORWARD_USAGE.md](docs/FORWARD_USAGE.md) — loading, forward shapes, pitfalls
- [docs/PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md) — which reported number comes from which artifact
- [docs/checkpoint_spec.json](docs/checkpoint_spec.json) — machine-readable checkpoint spec

This repository is the pre-training code only. Downstream policy evaluation
uses third-party harnesses (CortexBench, R3M, LIBERO, RVT-2, ACT), as cited in
the paper.

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

Apache-2.0. See [LICENSE](LICENSE).
