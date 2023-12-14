# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from vision_transformer import VisionTransformerCustom
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
# from mmcls.models import RedNet
from torchmetrics import StructuralSimilarityIndexMeasure
import torchvision
import torch.nn.functional as F
import random
import numpy as np

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]

class embed_conv(nn.Module):
    def __init__(self, embed_dim):
        super(embed_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(True),
                                  nn.Conv2d(64,64,3,1,1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(True))
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=8, stride=8)
    def forward(self, x):
        x = self.proj(self.conv(x))
        x = x.flatten(2).transpose(1,2)
        return x

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = 14*14
        del self.patch_embed
        self.conv_embed = embed_conv(self.embed_dim)
        # was 3, now 2 because no involution teacher
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_conv = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        self.head_conv.apply(self._init_weights)
        self.conv_embed.apply(self.init_weight)
        self.distillation_loss = DistillationLoss(
            s_dim = 192,
            t_dim = 3024,
            alpha = 2.0
        )

    def init_weight(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x_conv = self.conv_embed(x) # B N C

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        conv_token = x_conv.mean(1, keepdim=True)
        x = torch.cat((cls_tokens, conv_token, x_conv), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0], x[:, 1], x[:, 2:]

    def forward(self, x):
        x, x_conv, _ = self.forward_features(x)
        x = self.head(x)
        x_conv = self.head_conv(x_conv)
        if self.training:
            return x, x_conv
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_conv) / 2


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

class DistillationLoss(nn.Module):
    """
    Args:
        s_dim: the dimension of student's feature
        t_dim: the dimension of teacher's feature
    """
    def __init__(self, s_dim, t_dim, alpha):
        super(ITLoss, self).__init__()
        self.s_dim = s_dim
        self.t_dim = t_dim
        self.alpha = alpha

        self.embed = nn.Linear(s_dim, t_dim)
        self.bn_s = torch.nn.BatchNorm1d(t_dim, eps=0.0, affine=False)
        self.bn_t = torch.nn.BatchNorm1d(t_dim, eps=0.0, affine=False)

    def forward_simple(self, z_s, z_t):
        f_s = z_s
        f_t = z_t

        # must reshape the transformer repr
        b = f_s.shape[0]
        f_s = f_s.transpose(1, 2).view(b, -1, 14, 14)
        f_s = self.embed(f_s)

        f_s = F.normalize(f_s, dim=1)
        f_t = F.normalize(f_t, dim=1)

        return F.mse_loss(f_s, f_t)

    def forward(self, z_s, z_t):
        f_s = z_s
        f_t = z_t
        f_s = self.embed(f_s)
        n, d = f_s.shape

        f_s_norm = self.bn_s(f_s)
        f_t_norm = self.bn_t(f_t)

        c_st = torch.einsum('bx,bx->x', f_s_norm, f_t_norm) / n
        c_diff = c_st - torch.ones_like(c_st)

        c_diff = torch.abs(c_diff)
        c_diff = c_diff.pow(4.0)
        eps = 1e-5
        loss = torch.log(c_diff.sum() + eps)
        return loss