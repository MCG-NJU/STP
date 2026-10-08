"""
Minimal loader for the STP release on HuggingFace.

    import torch
    from loader import STPEncoder

    enc = STPEncoder.from_pretrained('yangjiange/STP')
    feats = enc.encode(images)      # (N, 3, 224, 224) -> (N, 768)

Or straight from the hub without this file:

    import torch, models_stp
    from huggingface_hub import hf_hub_download

    path = hf_hub_download('yangjiange/STP', 'stp_vitb.pth')
    model = models_stp.mae_vit_base_patch16()
    from util.ckpt_io import load_model_state
    state, _ = load_model_state(path)
    model.load_state_dict(state, strict=True)
    model.eval()
"""
import os

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class STPEncoder:
    """Frozen STP encoder: images in, [CLS] representations out.

    Only the encoder is used. The spatial and temporal decoders exist for
    pre-training and are irrelevant once you are training a policy.
    """

    def __init__(self, model, device='cpu', image_size=224):
        self.model = model
        self.device = device
        self.image_size = image_size

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, repo_id_or_path='yangjiange/STP', filename='stp_vitb.pth',
                        device=None, repo_root=None):
        import models_stp

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.isfile(repo_id_or_path):
            path = repo_id_or_path
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo_id_or_path, filename=filename,
                                   cache_dir=None)

        import sys as _sys, os as _os
        _root = repo_root or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from util.ckpt_io import load_model_state
        state, _ = load_model_state(path)

        if repo_root and repo_root not in os.sys.path:
            os.sys.path.insert(0, repo_root)

        model = models_stp.mae_vit_base_patch16()
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        return cls(model, device)

    # ------------------------------------------------------------------
    def preprocess(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def encode(self, images, batch_size=32):
        """PIL images / numpy arrays / a tensor -> (N, 768) numpy array."""
        if isinstance(images, torch.Tensor):
            out = []
            for i in range(0, len(images), batch_size):
                out.append(self.model.forward_features(
                    images[i:i + batch_size].to(self.device)).float().cpu())
            return torch.cat(out, 0).numpy()

        if isinstance(images, Image.Image):
            images = [images]
        feats = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.preprocess(im) for im in images[i:i + batch_size]])
            feats.append(self.model.forward_features(batch.to(self.device)).float().cpu())
        return torch.cat(feats, 0).numpy()
