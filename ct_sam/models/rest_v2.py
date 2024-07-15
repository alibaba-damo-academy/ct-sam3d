# ------------------------------------------------------------
# Copyright (c) VCU, Nanjing University.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by Qing-Long Zhang
# ------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from .common import LayerNorm3d


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.dim = dim

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                dim,
                dim,
                kernel_size=sr_ratio + 1,
                stride=sr_ratio,
                padding=sr_ratio // 2,
                groups=dim,
            )
            self.sr_norm = nn.LayerNorm(dim, eps=1e-6)

        # self.up = nn.Sequential(
        #     nn.Conv3d(dim, sr_ratio * sr_ratio * sr_ratio * dim, kernel_size=3, stride=1, padding=1, groups=dim),
        #     nn.PixelShuffle(upscale_factor=sr_ratio)
        # )
        self.up_norm = nn.LayerNorm(dim, eps=1e-6)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, D, H, W).contiguous()
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x = self.sr_norm(x)

        kv = (
            self.kv(x)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()

        identity = (
            v.transpose(-1, -2)
            .reshape(B, C, D // self.sr_ratio, H // self.sr_ratio, W // self.sr_ratio)
            .contiguous()
        )
        identity = (
            F.interpolate(identity, (D, H, W), mode="trilinear", align_corners=False)
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )
        x = self.proj(x + self.up_norm(identity))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, D, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W))  # pre_norm
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class Stem(nn.Module):
    def __init__(self, in_dim=1, out_dim=96, patch_size=2):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.proj = nn.Conv3d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()  # BCDHW -> BNC
        x = self.norm(x)
        D, H, W = D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]
        return x, (D, H, W)


class ConvStem(nn.Module):
    def __init__(self, in_ch=1, out_ch=96, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        stem = []
        in_dim, out_dim = in_ch, out_ch // 2
        for i in range(2):
            stem.append(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm3d(out_dim))
            stem.append(nn.ReLU(inplace=True))
            in_dim, out_dim = out_dim, out_dim * 2

        stem.append(nn.Conv3d(in_dim, out_ch, kernel_size=1, stride=1))
        self.proj = nn.Sequential(*stem)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2).contiguous()  # BCDHW -> BNC
        x = self.norm(x)
        D, H, W = D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]
        return x, (D, H, W)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, out_ch=96, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        if patch_size == 1:
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        else:
            self.proj = nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=patch_size + 1,
                stride=patch_size,
                padding=patch_size // 2,
            )

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2).contiguous()  # BCDHW -> BNC
        x = self.norm(x)
        D, H, W = D // self.patch_size[0], H // self.patch_size[0], W // self.patch_size[1]
        return x, (D, H, W)


class ResTV2(nn.Module):
    def __init__(
        self,
        in_chans=1,
        embed_dims=[96, 192, 384, 768],
        num_heads=[1, 2, 4, 8],
        drop_path_rate=0.0,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        img_size=64,
        out_chans=256,
    ):
        super().__init__()
        self.depths = depths
        self.img_size = img_size

        self.stem = ConvStem(in_chans, embed_dims[0], patch_size=4)
        self.patch_2 = PatchEmbed(embed_dims[0], embed_dims[1], patch_size=2)
        self.patch_3 = PatchEmbed(embed_dims[1], embed_dims[2], patch_size=2)
        self.patch_4 = PatchEmbed(embed_dims[2], embed_dims[3], patch_size=2)
        
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList(
            [
                Block(embed_dims[0], num_heads[0], sr_ratios[0], dpr[cur + i])
                for i in range(depths[0])
            ]
        )

        cur += depths[0]
        self.stage2 = nn.ModuleList(
            [
                Block(embed_dims[1], num_heads[1], sr_ratios[1], dpr[cur + i])
                for i in range(depths[1])
            ]
        )

        cur += depths[1]
        self.stage3 = nn.ModuleList(
            [
                Block(embed_dims[2], num_heads[2], sr_ratios[2], dpr[cur + i])
                for i in range(depths[2])
            ]
        )

        cur += depths[2]
        self.stage4 = nn.ModuleList(
            [
                Block(embed_dims[3], num_heads[3], sr_ratios[3], dpr[cur + i])
                for i in range(depths[3])
            ]
        )

        self.channel_mapper1 = nn.Conv3d(embed_dims[0], out_chans // 2, kernel_size=1)
        self.channel_mapper2 = nn.Conv3d(embed_dims[1], out_chans // 2, kernel_size=1)
        self.channel_mapper3 = nn.Conv3d(embed_dims[2], out_chans // 2, kernel_size=1)
        self.channel_mapper4 = nn.Conv3d(embed_dims[3], out_chans, kernel_size=1)

    def forward(self, x):
        out = []
        B, _, D, H, W = x.shape
        x, (D, H, W) = self.stem(x)
        # stage 1
        for blk in self.stage1:
            x = blk(x, D, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, D, H, W).contiguous()  # [B, -1, D//4, H//4, W//4]
        out.append(self.channel_mapper1(x))
        
        # stage 2
        x, (D, H, W) = self.patch_2(x)
        for blk in self.stage2:
            x = blk(x, D, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, D, H, W).contiguous()  # [B, -1, D//8, H//8, W//8]
        out.append(self.channel_mapper2(x))

        # stage 3
        x, (D, H, W) = self.patch_3(x)
        for blk in self.stage3:
            x = blk(x, D, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, D, H, W).contiguous()  # [B, -1, D//16, H//16, W//16]
        out.append(self.channel_mapper3(x))
        
        # stage 4
        x, (D, H, W) = self.patch_4(x)
        for blk in self.stage4:
            x = blk(x, D, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, D, H, W).contiguous()  # [B, -1, D//32, H//32, W//32]
        out.append(self.channel_mapper4(x))

        return out


def restv2_tiny(pretrained=False, **kwargs):  # 82.3|4.7G|24M -> |3.92G|30.37M   4.5G|30.33M
    model = ResTV2(embed_dims=[96, 192, 384, 768], depths=[1, 2, 6, 2], **kwargs)
    return model


def restv2_small(pretrained=False, **kwargs):  # 83.6|7.0G|35M   -> |5.78G|40.94M
    model = ResTV2(embed_dims=[96, 192, 384, 768], depths=[1, 2, 12, 2], **kwargs)
    return model


def restv2_base(pretrained=False, **kwargs):  # 84.4|10.2G|52M -> |7.25G|55.75M
    model = ResTV2(embed_dims=[96, 192, 384, 768], depths=[1, 3, 16, 3], **kwargs)
    return model


def restv2_large(pretrained=False, **kwargs):  # 85.3|39.6|218M -> |14.09G|98.61M
    model = ResTV2(
        num_heads=[2, 4, 8, 16], embed_dims=[128, 256, 512, 1024], depths=[2, 3, 16, 2], **kwargs
    )
    return model
