# STP forward 使用说明

三件事分开讲：**预训练 forward**（双解码器，出 loss）、**特征提取 forward**（冻结编码器，出 `[CLS]`）、**权重加载**。

---

## 0. 依赖

```
torch>=1.13        # zip 格式 checkpoint
torchvision
timm==0.3.2        # 参考实现 pin 的版本，别升
numpy, Pillow
tensorboard        # 只在训练时要
decord             # 只在 video backend / 抽帧时要
```

权重：`0.95_aug0.8.pth`，**torch 1.13+ 的 zip 格式**（不是裸 pickle），1.75 GB 里同时装着 `model` / `optimizer` / `epoch` / `scaler` / `args` 五个 key。只取 `model`。

---

## 1. 验证权重

不装 torch 也能验（`tools/verify_ckpt_params.py` 自带 torch stub，直接读 zip 里的 pickle）：

```
python tools/verify_ckpt_params.py 0.95_aug0.8.pth
```

预期输出：

```
tensors on disk : 434
tensors expected: 434
parameters      : 146.34 M
RESULT: 434/434 tensors matched
RESULT: OK -- strict=True load will succeed
```

装了 torch + timm 的话，用真实模型再验一次：

```
python -c "import torch, models_stp; m=models_stp.mae_vit_base_patch16(); \n  m.load_state_dict(torch.load('0.95_aug0.8.pth', map_location='cpu')['model'], strict=True); \n  print('434/434 OK')"
```

---

## 2. 加载权重（Python 里）

```python
import torch
import models_stp

model = models_stp.mae_vit_base_patch16(
    norm_pix_loss=True,
    mask_ratio_current=0.75,
    mask_ratio_future=0.95,
)

ckpt = torch.load('0.95_aug0.8.pth', map_location='cpu')
model.load_state_dict(ckpt['model'], strict=True)   # 434/434，可以 strict

print(ckpt['epoch'])        # 49
print(ckpt['args'])         # 预训练超参（见下）
model.eval()
```

`ckpt['args']` 里的关键值（我实测读出来的）：

```python
{'clip_model': 'ViT-B/16', 'batch_size': 64, 'epochs': 50, 'warmup_epochs': 5,
 'mask_ratio': 0.75, 'norm_pix_loss': True, 'blr': 0.00015,
 'weight_decay': 0.05, 'world_size': 64, 'seed': 0,
 'model': 'mae_vit_base_patch16', 'input_size': 224}
```

> 注意两点：
> 1. `mask_ratio_future` 不在 `args` 里（它是模型里的硬编码常量），用 0.95。

---

## 3. 预训练 forward（出 loss）

```python
# current_frame: (N, 3, 224, 224) 时刻 t 的帧
# future_frame : (N, 3, 224, 224) 时刻 t+16 的帧
loss, pred_c, pred_f, mask_c = model(current_frame, future_frame)
```

数据形状与语义：

| 张量 | 形状 | 含义 |
|---|---|---|
| `current_frame` | `(N, 3, 224, 224)` | 当前帧，**75% 掩码** |
| `future_frame` | `(N, 3, 224, 224)` | 未来帧，**95% 掩码**，也是 condition 来源 |
| `loss` | 标量 | `MSE(Î_c, I_c) + MSE(Î_f, I_f)`，只在被遮的 patch 上算 |
| `pred_c` | `(N, 196, 768)` | 空间解码器对当前帧被遮 patch 的预测（patch16 → 196 个 patch，每个 `16*16*3=768` 维） |
| `pred_f` | `(N, 196, 768)` | 时序解码器对未来帧被遮 patch 的预测 |
| `mask_c` | `(N, 196)` | 当前帧掩码，1=被遮，0=保留 |

`--norm_pix_loss` 为 True 时，重建目标是**逐 patch 标准化**后的像素，`pred_c` 也在这个空间里，别直接当图像看（要可视化先反标准化）。

内部数据流（对应论文 Eq.1–3）：

```
I_c ──patch_embed──▶ mask(75%) ──▶ ViT×12 ──▶ Z_c ─┬─▶ spatial decoder(8 blk) ──▶ Î_c
                                                    │
I_f ──patch_embed──▶ mask(95%) ──▶ ViT×12 ──▶ Z_f ──┴─▶ temporal decoder(8 blk, Z_c 作 K/V) ──▶ Î_f
```

关键点：**`Z_c` 在时序解码器里只做 key/value，且不回写**——当前帧表示空间不能被未来帧改写，这是消融里最要紧的一条（`self-cross` 63.7 vs `joint-self` 59.8）。

---

## 4. 特征提取 forward（下游策略用这个）

下游**冻结编码器、丢掉两个解码器**，取 `[CLS]`：

```python
feat = model.forward_features(imgs)      # (N, 768)，已带 @torch.no_grad()
```

这和 VC-1 / R3M 的用法一致：多视角/多帧特征在 channel 维 concat，再拼上本体感受状态，喂给策略网络（单任务 MLP / LIBERO transformer / RLBench RVT-2 / 真机 ACT）。

命令行版：

```
python tools/inference.py --ckpt 0.95_aug0.8.pth \
    --input frame.png --mode features
```

## 5. 跑一次完整 forward（自检）

```
python tools/inference.py --ckpt 0.95_aug0.8.pth \
    --input frame_t.png \
    --input-future frame_t16.png \
    --mode predict --output-dir out/
```

预期输出形态：

```
[stp] model=mae_vit_base_patch16  params=146.34M  (encoder 85.65M, decoders 60.54M)
[stp] loaded 434 tensors from 0.95_aug0.8.pth
[stp] forward -> loss=<数值>
[stp]   pred_spatial  (1, 196, 768)   masked patches 147/196
[stp]   pred_temporal (1, 196, 768)
```

`147/196 = 75%`，`186/196 ≈ 95%`——掩码数对得上就说明 forward 走通了。

---

## 6. 从零预训练

```
torchrun --nproc_per_node=8 main_pretrain_stp.py \
    --manifest /data/ego4d/annotations/egoclip_AVG3s.csv \
    --data_root /data/ego4d/ego4d_256_egoclip_AVG3s_img \
    --backend frame_dir --frame_interval 16 \
    --output_dir /exp/stp_vitb \
    --batch_size 64 --accum_iter 8 --epochs 50 --warmup_epochs 5 \
    --blr 1.5e-4 --weight_decay 0.05 --aug_scale_min 0.8
```

有效 batch = `64 * 8 * 8 = 4096`，和论文 Tab.5 一致。

数据准备（把 EgoClip 的 mp4 抽成 PNG 帧，因为采样是按绝对帧号 seek 的）：

```
python tools/prepare_ego4d_frames.py \
    --manifest /data/ego4d/annotations/egoclip_AVG3s.csv \
    --clip_root /data/ego4d/ego4d_256_egoclip_AVG3s \
    --output_root /data/ego4d/ego4d_256_egoclip_AVG3s_img \
    --workers 16 --shard 0 --num_shards 4
```

---

## 7. 几个容易踩的点

1. **`timm==0.3.2`**。参考代码里写着 `assert timm.__version__ == "0.3.2"`，新版 timm 的 `Block` 参数签名变了，直接报错。
2. **两帧必须用同一个随机种子做增强**。`datasets_egoclip.py` 里对 `img_c` / `img_f` 用同一个 `seed`，保证 crop 区域和翻转一致——否则时序预测的对应关系就被破坏成两张无关图了。
3. **`mask_ratio_future` 是模型属性，不是命令行参数**（在原始实现里硬编码在 `forward` 里）。这份 clean 版拆成了构造参数，默认 0.95。
4. **别用 `norm_pix_loss=False` 去加载**这份权重做可视化——权重是按 normalized target 训的，看起来会一片灰。
5. **官方权重只有 ViT-B**。ViT-L 的行（Tab.1/Tab.2 的 78.4/58.6）论文里是 post-pre-training 之后的结果，没有随这份材料发布。
6. **语言分支不要接**。Tab.4(b) 明写着加语言反而掉分（63.1 vs 63.7），且这份权重里根本没有语言参数。论文的结论就是「单模态自监督才是正解」。
