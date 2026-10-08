# Paper -> code map

Which code produces which number in the paper, and where the released
checkpoint sits in that map.

Paper: *Spatiotemporal Predictive Pre-training for Robotic Motor Control*,
*International Journal of Computer Vision* (2026), doi:[10.1007/s11263-026-02948-3](https://link.springer.com/article/10.1007/s11263-026-02948-3).
Supplementary: CVPR 2025 submission #11223.

---

## The released checkpoint

`0.95_aug0.8.pth` is the model behind the **`STP` rows** —
"STP / Ego" in Tab. 1, "STP" in Tab. 2, "STP" in Tab. 3, and the
`STP (EgoClip)` row of the appendix comparison table.

Its ablations in Tab. 4 read from the left column as:

| Tab. 4 row | value in checkpoint |
|---|---|
| (a) `rho_c = 75%`, predict = yes | `mask_ratio_current = 0.75` |
| (b) `95%` future mask, no language | `mask_ratio_future = 0.95`, no language params |
| (c) 8 `self-cross` decoder blocks | `decoder_blocks_2` = `DecoderBlock` (self + cross) |
| (d) frame interval 16 | sampling interval 16 |

`ckpt['args']` confirms the rest: `blr` 1.5e-4, `weight_decay` 0.05,
`epochs` 50, `warmup_epochs` 5, `norm_pix_loss` True, `world_size` 64,
and `ckpt['epoch'] == 49` (0-indexed final epoch).

Note the augmentation: the checkpoint's directory name carries `aug0.8`, and
the recipe correspondingly uses `RandomResizedCrop(0.8, 1.0)`. Both are
reproduced here as `--aug_scale_min 0.8`.

---

## Tab. 1 — single-task simulation

The released checkpoint is the `STP Ego` line (weighted average **63.7**).
The other lines are separate artifacts, not in this checkpoint:

| Tab. 1 line | how it is obtained |
|---|---|
| `MAE Ego` (59.6) | same data/epochs, MAE recipe only — the ablation baseline |
| `STP Ego+I` (64.2) | `STP Ego` additionally initialised from ImageNet-MAE (hybrid pre-training) |
| `MAE (Post PT)` (72.5) | task-data post-pre-training on top of `MAE Ego` |
| `STP (Post PT)` (76.4) | task-data post-pre-training on top of `STP Ego` |
| `STP (E2E FT)` (62.9) | encoder unfrozen during policy training |
| `STP Demo` (51.8) | pre-trained on task demonstrations only |
| `STP-L/16 (Post PT)` (78.4) | ViT-L variant, post-pre-trained — **not released** |

Hybrid and post-pre-training are adaptation procedures applied to a
pre-trained encoder; they do not produce new base checkpoints.

## Tab. 2 — multi-task simulation

`LIBERO-LONG` 41.3 ± 0.3 and `RLbench` 51.3 ± 1.4, both from the `STP` (ViT-B)
column. The `STP-L/16` column (36.7 / 58.6) needs the ViT-L encoder, which is
not released.

## Tab. 3 — real-world

Five tasks, standard environment (upper number) and unseen-with-distractors
(lower): weighted average 49.3 and 46.3. Two leader arms teleoperating two
follower arms, 50 demos per task, ACT policy. The demonstration videos in
`stp_supp/real-world demos/` are these five tasks.

## Tab. 4 — ablation on the released model

All rows are the same architecture with one thing changed, evaluated on
single-task sim (weighted average over Meta-World, Franka-Kitchen, DMControl,
Adroit, Trifinger).

**(a) current-frame masking** — the spatial branch

| `rho_c` | predict masked patches | WA |
|---|---|---|
| 75% | yes | **63.7** |
| 75% | no | 57.4 |
| 50% | yes | 59.0 |
| 0% | — | 57.0 |

**(b) temporal conditioning**

| condition | WA |
|---|---|
| language only | 55.4 |
| 95% future mask | **63.7** |
| 90% future mask | 63.4 |
| language (encoder) + 95% | 63.1 |
| language (decoder) + 95% | 60.9 |

**(c) temporal decoder**

| decoder | WA |
|---|---|
| 8 x joint-self | 59.8 |
| 12 x joint-self (param-matched) | 59.1 |
| 8 x self-cross | **63.7** |

The 3.9-point gap between the two 8-block designs is the single largest
architectural effect in the paper, and it is a one-line difference: whether
`Z_c` can be updated by the future-frame query.

**(d) frame interval**

| interval | WA |
|---|---|
| 8 | 61.3 |
| **16** | **63.7** |
| 24 | 62.5 |
| random 8-24 | 60.8 |

## Appendix-only comparisons

| method | WA |
|---|---|
| STP (EgoClip) | 63.7 |
| VideoMAE (EgoClip, 4 frames) | 52.6 |
| LIV (ResNet-50) | 58.1 |
| MPI (ViT-base, stripped to `[CLS]`) | 58.7 |

Reinforcement learning setup (DrQ-v2, max reward over 200k steps):
Panda-Door 95.6 vs VC-1 88.8; Panda-TwoArmPegInHole 130.7 vs 123.1.

Stability: three reruns give 63.9 ± 0.3 (STP-B) against 58.7 ± 0.3 (MAE-B),
7350 episodes in total.

Post-pre-training also fixes the loss weighting at 1:1 (temporal : spatial);
the alternative ratios score 54.7 (3:1) and 55.2 (1:3) against 57.4 (1:1) on
the five Franka-Kitchen tasks.

---

## What is in this repository

| component | file |
|---|---|
| released model | `models_stp.py` |
| pre-training entry | `main_pretrain_stp.py` |
| training loop | `engine_pretrain_stp.py` |
| EgoClip sampling | `datasets_egoclip.py` |
| checkpoint loading, forward | `tools/load_stp.py` |
| checkpoint verification | `tools/verify_ckpt_params.py` |
| encoder-only export | `tools/export_encoder.py` |
| frame extraction | `tools/prepare_ego4d_frames.py` |

## What is not in this repository

- **Downstream policy training.** Evaluation used third-party code as published
  by prior work: the [VC-1 / CortexBench](https://github.com/facebookresearch/eai-vc/tree/main/cortexbench)
  harness for Meta-World, DMControl, Adroit and Trifinger, and the
  [R3M](https://github.com/facebookresearch/r3m/tree/eval/evaluation) harness
  for Franka-Kitchen; LIBERO-LONG uses the official transformer policy and
  RLBench uses [RVT-2](https://github.com/NVlabs/RVT). ACT was used for the
  real-world tasks. Reproducing a Tab. 1 or Tab. 3 number means dropping the
  frozen encoder into those harnesses, not running anything here.
- **Post-pre-training.** The adaptation runs (Tab. 1, `Post PT` rows) use the
  same model and loss with task data and the schedules in the appendix tables.
- **The ViT-L encoder.** Tab. 1 and Tab. 2 `STP-L/16` columns are unreleased.
- **The language-fusion variants.** Eight `models_mae_lang_future_*` variants
  existed in the research codebase; they are deliberately omitted, since the
  ablation shows language hurts and the released weights contain no language
  parameters.
