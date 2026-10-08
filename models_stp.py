# Copyright (c) 2024 Jiange Yang, Nanjing University / Microsoft Research
# All rights reserved.
#
# STP: Spatiotemporal Predictive Pre-training for Robotic Motor Control
# IJCV 2026: https://link.springer.com/article/10.1007/s11263-026-02948-3
#
# This file is a clean re-implementation of the pre-training model. It is
# parameter-name compatible with the released checkpoint
# `lang_0.95_aug0.8_no_lang.pth` (434 tensors, verified strict=True).
#
# Reference implementations:
#   MAE  : https://github.com/facebookresearch/mae
#   timm : https://github.com/rwightman/pytorch-image-models
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn

# `timm==0.3.2` (the version this model was trained with) imports
# `container_abcs` from `torch._six`, which PyTorch removed in 2.0. Installing
# the shim before importing timm keeps that pinned version working on modern
# PyTorch without editing site-packages.
import util.torch_six_compat  # noqa: F401  (must precede the timm import)

from timm.models.vision_transformer import Block, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed


class CrossAttention(nn.Module):
    """Cross attention with a *frozen* key/value context.

    In STP's temporal decoder the current-frame feature `context` acts as
    key/value and is never updated, and the future-frame query cannot write
    back into the current-frame representation space. This asymmetry is what
    keeps the encoder's image representation free of future-frame leakage.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        M = context.shape[1]

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    """Temporal decoder block: self-attn -> cross-attn(context) -> FFN."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = _SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, context):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.query_norm(x), self.context_norm(context))
        x = x + self.mlp(self.norm2(x))
        return x


class _SelfAttention(nn.Module):
    """Plain multi-head self attention. Named so the released checkpoint's
    `self_attn.qkv.*` / `self_attn.proj.*` keys load exactly."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MaskedAutoencoderViT(nn.Module):
    """STP: one shared ViT encoder + decoupled spatial / temporal decoders.

    Pre-training samples a frame pair (current frame I_c, future frame I_f)
    from a clip and applies *asymmetric* masking:

        spatial  branch: I_c masked at mask_ratio_current (75%) -> spatial decoder
                         predicts the removed patches of I_c.
        temporal branch: I_f masked at mask_ratio_future  (95%) -> temporal decoder
                         predicts the removed patches of I_f, conditioned on Z_c.

    Only the encoder is kept for downstream policy learning; both decoders are
    discarded. The encoder is a plain image ViT with no cross-frame attention.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_current=0.75, mask_ratio_future=0.95):
        super().__init__()

        # ------------------------------------------------------------------
        # encoder
        # ------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # ------------------------------------------------------------------
        # dual decoders
        # ------------------------------------------------------------------
        self.decoder_embed_1 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_2 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos

        # spatial decoder: standard transformer encoder blocks
        self.decoder_blocks_1 = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        # temporal decoder: self-attention + cross-attention over the current frame
        self.decoder_blocks_2 = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                         qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_1 = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.decoder_pred_2 = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio_current = mask_ratio_current
        self.mask_ratio_future = mask_ratio_future

        self.initialize_weights()

    # ----------------------------------------------------------------------
    # init
    # ----------------------------------------------------------------------
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5),
                                                    cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def patchify(self, imgs):
        """imgs: (N, 3, H, W) -> (N, L, patch_size**2 * 3)"""
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

    def unpatchify(self, x):
        """x: (N, L, patch_size**2 * 3) -> (N, 3, H, W)"""
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], 3, h * p, h * p))

    def random_masking(self, x, mask_ratio):
        """Per-sample random masking. Returns kept tokens, binary mask (1=removed), restore idx."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    # ----------------------------------------------------------------------
    # encoder / decoders
    # ----------------------------------------------------------------------
    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def _assemble(self, x, ids_restore, decoder_embed):
        """Project, re-insert mask tokens in the original order, add decoder pos embed."""
        x = decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)          # drop cls
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)                    # restore cls
        return x + self.decoder_pos_embed

    def forward_decoder_spatial(self, x, ids_restore):
        x = self._assemble(x, ids_restore, self.decoder_embed_1)
        for blk in self.decoder_blocks_1:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred_1(x)[:, 1:, :]

    def forward_decoder_temporal(self, x, context, ids_restore):
        x = self._assemble(x, ids_restore, self.decoder_embed_2)
        context = self.decoder_embed_2(context)                    # Z_c stays fixed
        for blk in self.decoder_blocks_2:
            x = blk(x, context)
        x = self.decoder_norm(x)
        return self.decoder_pred_2(x)[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                                   # [N, L]
        return (loss * mask).sum() / mask.sum()                    # only removed patches

    # ----------------------------------------------------------------------
    # forward (pre-training)
    # ----------------------------------------------------------------------
    def forward(self, current_frame, future_frame):
        """One pre-training step.

        current_frame: (N, 3, H, W) the frame at time t
        future_frame : (N, 3, H, W) the frame at time t + interval

        returns (loss, pred_spatial, pred_temporal, mask_spatial)
        """
        latent_c, mask_c, ids_c = self.forward_encoder(current_frame, self.mask_ratio_current)
        latent_f, mask_f, ids_f = self.forward_encoder(future_frame, self.mask_ratio_future)

        pred_c = self.forward_decoder_spatial(latent_c, ids_c)
        pred_f = self.forward_decoder_temporal(latent_f, latent_c, ids_f)

        loss = self.forward_loss(current_frame, pred_c, mask_c) \
             + self.forward_loss(future_frame, pred_f, mask_f)
        return loss, pred_c, pred_f, mask_c

    # ----------------------------------------------------------------------
    # frozen-encoder feature extraction (this is what downstream policies use)
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def forward_features(self, imgs):
        """Full (unmasked) image -> [CLS] representation, shape (N, embed_dim).

        This is the interface used by downstream policy learning: the encoder
        is frozen and the [CLS] token is taken as the global visual feature.
        """
        x = self.patch_embed(imgs) + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


def mae_vit_base_patch16_dec512d8b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def mae_vit_large_patch16_dec512d8b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


# recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b   # ViT-B, released checkpoint
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b
