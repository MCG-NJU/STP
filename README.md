# STP: Spatiotemporal Predictive Pre-training for Robotic Motor Control

Clean, runnable implementation of STP, with a verified loader for the released
pre-trained weights.

> Jiange Yang, Bei Liu, Jianlong Fu, Bocheng Pan, Gangshan Wu, Limin Wang.
> **Spatiotemporal Predictive Pre-training for Robotic Motor Control.**
> [arXiv:2403.05304](https://arxiv.org/abs/2403.05304)

STP pre-trains a **plain image ViT** for robotic motor control from large-scale
egocentric video (Ego4D / EgoClip), with **no** language, action, depth or
teacher supervision. It jointly performs two masked-prediction tasks with a
shared encoder and **decoupled dual decoders**:

| branch | input | mask ratio | decoder | learns |
|---|---|---|---|---|
| spatial | current frame `I_c` | **75%** | transformer encoder blocks | content / geometry |
| temporal | future frame `I_f` | **95%** | self-attn + cross-attn block | motion / dynamics |

The temporal decoder uses `Z_c` (the current-frame feature) as a **fixed**
key/value — the future-frame query cannot write back into the current-frame
representation. That asymmetry is the single most important design choice
(`self-cross` 63.7 vs `joint-self` 59.8 weighted average).

At evaluation the encoder is **frozen**, both decoders are discarded, and the
`[CLS]` token is used as the visual representation for few-shot behavior
cloning.

---

## Checkpoint

`lang_0.95_aug0.8_no_lang.pth` — ViT-B/16 encoder + two 8-block decoders,
146.34 M parameters (encoder 85.65 M + spatial decoder 26.01 M + temporal decoder 34.43 M + shared 0.10 M), trained 50 epochs on EgoClip.

```python
ckpt = torch.load('lang_0.95_aug0.8_no_lang.pth', map_location='cpu')
ckpt.keys()      # dict_keys(['model', 'optimizer', 'epoch', 'scaler', 'args'])
ckpt['epoch']    # 49 (0-indexed final epoch)
```

The file is a **torch >= 1.13 zip-format** checkpoint. `ckpt['model']` holds
434 tensors; **all 434 match this architecture exactly** (`strict=True` loads
cleanly). Verify without even installing torch:

```
python tools/verify_ckpt_params.py lang_0.95_aug0.8_no_lang.pth
# RESULT: 434/434 tensors matched
# RESULT: OK -- strict=True load will succeed
```

Despite the `lang_` in its name, this checkpoint contains **no language
parameters** — it is the single-modality variant, which is also what the
ablation found best (adding language drops 63.7 -> 63.1).

---

## Install

```
pip install torch torchvision timm==0.3.2 numpy pillow
pip install tensorboard decord        # training / frame extraction only
```

`timm==0.3.2` matters: the reference code asserts it, and newer timm changed
the `Block` signature.

---

## Forward: pre-training

```python
import torch, models_stp

model = models_stp.mae_vit_base_patch16(
    norm_pix_loss=True, mask_ratio_current=0.75, mask_ratio_future=0.95)
model.load_state_dict(torch.load('lang_0.95_aug0.8_no_lang.pth')['model'], strict=True)
model.eval()

# I_c at time t, I_f at t+16; both (N, 3, 224, 224), ImageNet-normalised
loss, pred_c, pred_f, mask_c = model(current_frame, future_frame)
# loss:    scalar, MSE on masked patches of both frames
# pred_c:  (N, 196, 768)  spatial decoder output
# pred_f:  (N, 196, 768)  temporal decoder output
# mask_c:  (N, 196)       1 = removed (147/196 at 75%)
```

## Forward: frozen feature extraction (what downstream policies use)

```python
feat = model.forward_features(imgs)      # (N, 768), the [CLS] token
```

Match that against VC-1 / R3M usage: concatenate multi-view, multi-frame
features in the channel dimension together with proprioception, then train the
policy (MLP / transformer / RVT-2 / ACT). Or export a decoder-free ViT that
drops straight into existing `timm` backbone code:

```
python tools/export_encoder.py --ckpt lang_0.95_aug0.8_no_lang.pth --output stp_vitb_encoder.pth
```

## End-to-end self-check

```
python tools/load_stp.py --ckpt lang_0.95_aug0.8_no_lang.pth \
    --current samples/current_frame.png --future samples/future_frame.png
```

`samples/` holds a real frame pair (frames 0 and 16) pulled from the
supplementary material's real-world demo video.

See **[docs/FORWARD_USAGE.md](docs/FORWARD_USAGE.md)** for the full walkthrough,
shapes, and the pre-training/eval hyperparameters read back out of the
checkpoint. Machine-readable summary: [docs/checkpoint_spec.json](docs/checkpoint_spec.json).

---

## Pre-train from scratch

```
torchrun --nproc_per_node=8 main_pretrain_stp.py \
    --manifest /data/ego4d/annotations/egoclip_AVG3s.csv \
    --data_root /data/ego4d/ego4d_256_egoclip_AVG3s_img \
    --backend frame_dir --frame_interval 16 \
    --output_dir /exp/stp_vitb \
    --batch_size 64 --accum_iter 8 --epochs 50 --warmup_epochs 5 \
    --blr 1.5e-4 --weight_decay 0.05 --aug_scale_min 0.8
```

Effective batch size `64 * 8 * 8 = 4096`, matching the paper.

Prepare frames first (sampling seeks by absolute frame index, so decoding each
clip once up front is far cheaper than seeking into mp4s every step):

```
python tools/prepare_ego4d_frames.py \
    --manifest /data/ego4d/annotations/egoclip_AVG3s.csv \
    --clip_root /data/ego4d/ego4d_256_egoclip_AVG3s \
    --output_root /data/ego4d/ego4d_256_egoclip_AVG3s_img \
    --workers 16 --shard 0 --num_shards 4
```

---

## Release recipe

From the paper's appendix (and confirmed against `ckpt['args']`):

| | |
|---|---|
| optimizer | AdamW, `blr` 1.5e-4, wd 0.05, betas (0.9, 0.95), cosine decay |
| schedule | 50 epochs, 5 warmup, effective batch 4096 |
| augmentation | RandomResizedCrop **(0.8, 1.0)**, horizontal flip |
| frame sampling | interval **16** (fixed), alpha 0.3 for clips <= 60 frames |
| masks | current 75% (predict masked patches), future **95%** (as condition) |
| loss | `MSE(I_c) + MSE(I_f)`, 1:1 weight, per-patch normalised targets |
| encoder | ViT-B/16, 12 layers, 12 heads, 768, fixed sin-cos pos embed |
| decoders | 2 x 8 layers, 16 heads, 512, fixed sin-cos pos embed |

Selected ablations (weighted average over 5 single-task sim benchmarks):

| setting | WA |
|---|---|
| both branches, self-cross temporal decoder | **63.7** |
| joint-self temporal decoder (8 blocks) | 59.8 |
| joint-self (12 blocks, param-matched) | 59.1 |
| no spatial prediction on current frame | 57.4 |
| future frame not masked to 95% | — |
| language only, no future frame | 55.4 |
| language + 95% future frame | 63.1 |
| frame interval 8 / 16 / 24 | 61.3 / **63.7** / 62.5 |

vs. baselines on the same EgoClip data: MAE 59.6, VC-1 61.8, MPI 58.7,
LIV 58.1, VideoMAE 52.6.

---

## Layout

```
models_stp.py              model: shared encoder + dual decoders + forward_features
main_pretrain_stp.py       pre-training entry point
engine_pretrain_stp.py     training loop
datasets_egoclip.py        EgoClip frame-pair dataset
util/                      misc, lr_sched, pos_embed (from the MAE codebase)
tools/load_stp.py          load the checkpoint; run forward / extract features
tools/verify_ckpt_params.py  verify the 434-tensor match (no torch needed)
tools/export_encoder.py    strip the decoders, emit a plain ViT backbone
tools/prepare_ego4d_frames.py  mp4 -> PNG frame tree
docs/FORWARD_USAGE.md      detailed forward / loading walkthrough
```

## Notes on this release

This is a cleaned-up version of the research codebase used for the paper. Two
things in the original experiment tree were deliberately not carried over:

- The `models_mae_lang_future_*` family (8 language-fusion variants) is out of
  scope — language hurt performance and is not in the released weights.
- Dead branches and stale configs (a half-edited variant that crashes on
  construct, hard-coded `/mnt/my_output/...` paths, an `except Exception` that
  silently swallowed data errors) were removed rather than reproduced. The
  sampler's unbounded recursion on a missing file became a bounded retry.

Numerically, the model, the sampling rule, the augmentation, the masking ratios
and the loss are unchanged.

## License

Apache-2.0. See [LICENSE](LICENSE).

The `util/` helpers are STP's own implementations rather than copies of the MAE
utilities, so the whole repository is covered by a single permissive licence.
The sincos positional-embedding tables are verified bit-identical to the
reference implementation, and the learning-rate schedule matches it pointwise,
so this does not affect reproducibility.
