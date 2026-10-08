"""
Frozen-encoder inference with a released STP checkpoint.

Three things you can do with a pre-trained STP encoder:

  1. `features`  -- image(s) -> [CLS] visual representation. This is what
     downstream policy learning consumes; the encoder is frozen and both
     decoders are unused.
  2. `predict`   -- a frame pair -> the masked spatiotemporal predictions, i.e.
     what pre-training optimises. Useful to eyeball that a checkpoint is sane.
  3. `video`     -- frame features for every frame of a clip, at a stride.

Examples:
    # one image -> a 768-d vector
    python tools/inference.py --ckpt weights/lang_0.95_aug0.8_no_lang.pth \\
        --input samples/current_frame.png --mode features

    # a directory of images -> (N, 768)
    python tools/inference.py --ckpt weights/lang_0.95_aug0.8_no_lang.pth \\
        --input 'frames/*.png' --mode features --output feats.npy

    # a video -> (T, 768) per-frame features at stride 4
    python tools/inference.py --ckpt weights/lang_0.95_aug0.8_no_lang.pth \\
        --input clip.mp4 --mode video --stride 4 --output clip_feats.npy

    # a frame pair -> reconstructions written to disk
    python tools/inference.py --ckpt weights/lang_0.95_aug0.8_no_lang.pth \\
        --input samples/current_frame.png --input-future samples/future_frame.png \\
        --mode predict --output-dir out/

The checkpoint may be a local .pth/.pt, or a HuggingFace repo id such as
`MCG-NJU/STP` -- in which case the file is downloaded and cached via
`huggingface_hub`. Requires `pip install huggingface_hub` only for that path.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models_stp  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
DEFAULT_HF_FILENAME = 'lang_0.95_aug0.8_no_lang.pth'


# ---------------------------------------------------------------------------
# checkpoint resolution
# ---------------------------------------------------------------------------
def resolve_checkpoint(path):
    """Accept a local file, or a HuggingFace repo id / `repo_id::filename`."""
    if os.path.isfile(path):
        return path

    if path.endswith(('.pth', '.pt')):
        raise FileNotFoundError(f'checkpoint not found: {path}')

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            f'`{path}` is not a local file; to download from HuggingFace '
            f'install huggingface_hub (`pip install huggingface_hub`).') from e

    repo_id, _, filename = path.partition('::')
    filename = filename or DEFAULT_HF_FILENAME
    print(f'[stp] downloading {repo_id}/{filename} from HuggingFace ...')
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_model(ckpt_path, model_name='mae_vit_base_patch16', device='cpu',
               mask_ratio_current=0.75, mask_ratio_future=0.95, verbose=True):
    """Build the STP model and load the released pre-training weights."""
    model = models_stp.__dict__[model_name](
        norm_pix_loss=True,
        mask_ratio_current=mask_ratio_current,
        mask_ratio_future=mask_ratio_future)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    if verbose:
        n = sum(v.numel() for v in model.state_dict().values())
        print(f'[stp] loaded {len(state)} tensors ({n/1e6:.2f} M params) from {ckpt_path}')
        if isinstance(ckpt, dict) and 'epoch' in ckpt:
            print(f'[stp] checkpoint epoch = {ckpt["epoch"]}')
    return model


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def preprocess(img, size=224):
    """PIL image -> normalised (3, size, size) tensor."""
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def denormalise(x):
    """Undo the ImageNet normalisation for visualisation."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def read_frames(path, stride=1):
    """Read PIL frames from a video file (needs decord) or a directory/glob."""
    if os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS):
        from decord import VideoReader, cpu
        with open(path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
            idx = list(range(0, len(vr), stride))
            frames = vr.get_batch(idx).asnumpy()
        return [Image.fromarray(f) for f in frames], idx

    if os.path.isdir(path):
        files = sorted(p for p in glob.glob(os.path.join(path, '*'))
                       if p.lower().endswith(IMAGE_EXTS))
    else:
        files = sorted(glob.glob(path))
    files = files[::stride]
    return [Image.open(p).convert('RGB') for p in files], list(range(0, len(files)))


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_features(model, imgs, device, batch_size=32):
    """(N, 768) [CLS] features, the representation policies are trained on."""
    out = []
    for i in range(0, len(imgs), batch_size):
        batch = torch.stack([preprocess(im) for im in imgs[i:i + batch_size]]).to(device)
        out.append(model.forward_features(batch).float().cpu())
    return torch.cat(out, 0).numpy()


@torch.no_grad()
def run_predict(model, img_c, img_f, device, output_dir):
    """One pre-training forward: loss plus masked reconstructions."""
    xc = preprocess(img_c).unsqueeze(0).to(device)
    xf = preprocess(img_f).unsqueeze(0).to(device)
    loss, pred_c, pred_f, mask_c = model(xc, xf)

    n_patches = pred_c.shape[1]
    print(f'[stp] loss = {loss.item():.4f}')
    print(f'[stp] spatial  : {tuple(pred_c.shape)}, '
          f'masked {int(mask_c.sum().item())}/{n_patches} '
          f'({mask_c.float().mean().item()*100:.0f}%)')
    print(f'[stp] temporal : {tuple(pred_f.shape)}, '
          f'masked ~{int(n_patches * model.mask_ratio_future)}/{n_patches} '
          f'({model.mask_ratio_future*100:.0f}%)')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name, pred in (('spatial', pred_c), ('temporal', pred_f)):
            rec = model.unpatchify(pred)[0]
            # predictions live in per-patch normalised space (norm_pix_loss);
            # rescale per-image for a legible picture, not for measurement
            rec = rec - rec.mean()
            rec = rec / (rec.std() + 1e-6) * 0.2 + 0.5
            img = Image.fromarray((rec.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            path = os.path.join(output_dir, f'recon_{name}.png')
            img.save(path)
            print(f'[stp] wrote {path}')
        # also dump the ground-truth pair for side-by-side inspection
        for name, im in (('current', img_c), ('future', img_f)):
            im.resize((224, 224), Image.BICUBIC).save(os.path.join(output_dir, f'gt_{name}.png'))
    return loss.item()


def save_array(arr, path):
    if path is None:
        return
    if path.endswith('.npy'):
        np.save(path, arr)
    elif path.endswith('.pt') or path.endswith('.pth'):
        torch.save(torch.from_numpy(arr), path)
    else:
        np.savetxt(path, arr, delimiter=',')
    print(f'[stp] wrote {path}  shape={arr.shape}')


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser('STP inference',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument('--ckpt', required=True,
                   help='local .pth/.pt, or a HF repo id such as MCG-NJU/STP')
    p.add_argument('--input', required=True,
                   help='image, glob, directory, or video')
    p.add_argument('--input-future', default=None,
                   help='second frame for --mode predict')
    p.add_argument('--mode', default='features',
                   choices=['features', 'predict', 'video'])
    p.add_argument('--model', default='mae_vit_base_patch16')
    p.add_argument('--device', default=None, help='default: cuda if available')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--stride', type=int, default=1,
                   help='for --mode video: keep every Nth frame')
    p.add_argument('--output', default=None,
                   help='.npy / .pt / .txt destination for the feature array')
    p.add_argument('--output-dir', default=None,
                   help='destination for --mode predict reconstructions')
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = resolve_checkpoint(args.ckpt)
    model = load_model(ckpt_path, args.model, device,
                       mask_ratio_current=0.75, mask_ratio_future=0.95)
    print(f'[stp] device = {device}')

    if args.mode == 'predict':
        if not args.input_future:
            raise SystemExit('--mode predict needs --input-future')
        run_predict(model, Image.open(args.input).convert('RGB'),
                    Image.open(args.input_future).convert('RGB'), device, args.output_dir)
        return

    imgs, indices = read_frames(args.input, stride=args.stride)
    if not imgs:
        raise SystemExit(f'no images found at {args.input}')
    print(f'[stp] {len(imgs)} frame(s), stride {args.stride}')

    feats = run_features(model, imgs, device, args.batch_size)
    print(f'[stp] features: {feats.shape}  dtype={feats.dtype}  '
          f'mean={feats.mean():.4f}  std={feats.std():.4f}')

    if args.mode == 'video':
        # print a compact per-frame summary so a quick run is readable
        for k, i in enumerate(indices):
            v = feats[k]
            print(f'  frame {i:5d}  norm={np.linalg.norm(v):8.3f}  '
                  f'min={v.min():7.3f}  max={v.max():7.3f}')

    save_array(feats, args.output)


if __name__ == '__main__':
    main()
