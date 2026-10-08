"""
EgoClip frame-pair dataset for STP pre-training.

Each item is a (current_frame, future_frame, narration) triple sampled from one
EgoClip clip. The two frames are drawn a fixed interval apart, and -- crucially
-- are transformed with the *same* random seed, so they receive the identical
RandomResizedCrop region and horizontal flip. That keeps the spatial
correspondence between I_c and I_f intact, which the temporal prediction branch
depends on.

Two backends are supported:

  frame_dir backend (default, what the paper used)
      Pre-extracted PNG frames on disk:
          {root}/{video_uid}/{narration_source}_{idx}_clip/{frame_idx}.png
      See tools/prepare_ego4d_frames.py for how to build this tree.

  video backend
      Raw clip mp4s read on the fly with decord:
          {root}/{video_uid}/{narration_source}_{idx}_clip.mp4

Manifest: the EgoClip CSV, with columns
    video_uid, narration_source, narration_ind, clip_text, length
`length` is the clip length in frames.
"""
import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Interval between the current and the future frame. 16 is the value that won
# the frame-sampling ablation (Tab. 4d, 63.7 WA vs 61.3 for 8 and 62.5 for 24).
DEFAULT_FRAME_INTERVAL = 16

# The original sampler only applies its fixed-interval rule to clips longer
# than this many frames; shorter clips use the alpha rule below.
LONG_CLIP_THRESHOLD = 60
DEFAULT_ALPHA = 0.3


class EgoClipFrames(Dataset):
    def __init__(self, manifest_path, root, transform=None, backend='frame_dir',
                 frame_interval=DEFAULT_FRAME_INTERVAL, alpha=DEFAULT_ALPHA,
                 max_retries=8):
        self.root = root
        self.transform = transform
        self.backend = backend
        self.frame_interval = frame_interval
        self.alpha = alpha
        self.max_retries = max_retries

        with open(manifest_path, 'r', newline='') as fp:
            self.manifest = list(csv.DictReader(fp))

    def __len__(self):
        return len(self.manifest)

    # ------------------------------------------------------------------
    def _clip_id(self, clip):
        return '{}_{}_clip'.format(clip['narration_source'], clip['narration_ind'])

    def _sample_indices(self, vidlen):
        """Replicates the reference sampler exactly."""
        vidlen = min(vidlen, 10240 - 10)
        if vidlen > LONG_CLIP_THRESHOLD:
            start = np.random.randint(20, vidlen - 30)
            end = start + self.frame_interval
        else:
            start = np.random.randint(0, 2 + int(self.alpha * vidlen))
            end = np.random.randint(int((1 - self.alpha) * vidlen) - 2, vidlen)
        return start, end

    def _read_pair(self, clip, start_idx, end_idx):
        uid, clip_id = clip['video_uid'], self._clip_id(clip)
        if self.backend == 'frame_dir':
            d = os.path.join(self.root, uid, clip_id)
            a = Image.open(os.path.join(d, f'{start_idx}.png')).convert('RGB')
            b = Image.open(os.path.join(d, f'{end_idx}.png')).convert('RGB')
            return a, b

        if self.backend == 'video':
            from decord import VideoReader, cpu
            path = os.path.join(self.root, uid, clip_id + '.mp4')
            with open(path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
                frames = vr.get_batch([start_idx, end_idx]).asnumpy()
            return Image.fromarray(frames[0]), Image.fromarray(frames[1])

        raise ValueError(f'unknown backend: {self.backend}')

    # ------------------------------------------------------------------
    def __getitem__(self, i):
        """Scan forward on a missing/corrupt clip rather than giving up.

        The reference implementation recursed via `self.__getitem__(i + 100000)`
        on a 3.8 M-row manifest, which both risks blowing the stack and can spin
        forever on a small manifest. Here we try at most `max_retries` distinct
        indices -- the requested one first, then successive ones -- so a single
        bad clip degrades to its neighbour instead of killing the worker.
        """
        n = len(self.manifest)
        for attempt in range(min(self.max_retries, n)):
            clip = self.manifest[(i + attempt) % n]
            try:
                start_idx, end_idx = self._sample_indices(int(clip['length']))
                img_c, img_f = self._read_pair(clip, start_idx, end_idx)
            except (FileNotFoundError, OSError, IndexError, ValueError):
                continue

            if self.transform is None:
                return img_c, img_f, clip['clip_text']

            # Same seed for both frames: identical crop and flip.
            seed = random.randint(0, 2 ** 32)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            first = self.transform(img_c)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            second = self.transform(img_f)

            return first, second, clip['clip_text']

        raise RuntimeError(
            f'no readable clip among {min(self.max_retries, n)} candidates starting at index {i}')
