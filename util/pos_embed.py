"""
2D sine-cosine positional embeddings.

STP uses a *fixed* sincos table (the checkpoint stores it as a non-trainable
buffer), so these tables must be reproduced exactly at initialisation time.
"""
import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """2D sincos table for a `grid_size` x `grid_size` patch grid.

    Returns [grid_size**2, embed_dim], or [1 + grid_size**2, embed_dim] when
    `cls_token=True` (a zero row is prepended for the [CLS] token).
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)          # w first, so the flatten is row-major
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0, 'embed_dim must be even to split between h and w'

    # half of the dimensions encode the height, the other half the width
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sincos table: `pos` of shape (M,) -> (M, embed_dim)."""
    assert embed_dim % 2 == 0

    # geometric frequency ladder, then the outer product with the positions
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)
