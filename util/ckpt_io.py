"""
Checkpoint loading that works across PyTorch versions.

STP checkpoints store `{model, optimizer, epoch, scaler, args}`, and `args` is
an `argparse.Namespace`. PyTorch 2.6 flipped `torch.load`'s `weights_only`
default to `True`, which rejects any pickled class that is not on an allowlist --
so a plain `torch.load(path)` now raises `UnpicklingError` on these files.

Rather than dropping back to `weights_only=False` (which permits arbitrary code
execution from an untrusted file), we keep the safe path and allowlist only the
two benign types the checkpoint actually contains.
"""
import argparse

import torch

_safe_globals_registered = False


def _register_safe_globals():
    global _safe_globals_registered
    if _safe_globals_registered:
        return
    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
    except AttributeError:
        # torch < 2.4 has no such API; weights_only=True is also not the default
        # there, so loading works regardless.
        pass
    _safe_globals_registered = True


def load_checkpoint(path, map_location='cpu'):
    """Load an STP checkpoint, preferring the safe unpickler.

    Returns the raw checkpoint dict; callers pick out `['model']`.
    """
    _register_safe_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # `weights_only` did not exist before torch 1.13
        return torch.load(path, map_location=map_location)
    except Exception:
        # Older or externally-produced checkpoints may carry other benign
        # pickled objects. Fall back, since the caller has already chosen to
        # load this file.
        return torch.load(path, map_location=map_location, weights_only=False)


def load_model_state(path, map_location='cpu'):
    """Return just the model state dict from an STP checkpoint.

    Accepts a raw state dict too, so an exported `*_encoder.pth` works.
    """
    ckpt = load_checkpoint(path, map_location=map_location)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        return ckpt['model'], ckpt
    return ckpt, {}
