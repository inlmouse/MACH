# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn

logger = logging.getLogger("dinov3")

class ConditionalLayerNorm(nn.Module):
    def __init__(self, normalized_shape, cond_dim, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)  # 关闭默认 affine
        self.gamma_proj = nn.Linear(cond_dim, normalized_shape)   # 或直接用截取，如果 cond_dim == normalized_shape
        self.beta_proj  = nn.Linear(cond_dim, normalized_shape)
        # 初始化让它接近 identity
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, x, cond):
        # x: [B, C, H, W] 或 [B, L, C]
        # cond: [B, cond_dim] （你的 mean-pooled MRL 前C维 或投影后）
        
        out = self.ln(x)                   # 先标准 LN，无 affine
        
        gamma = self.gamma_proj(cond)      # [B, C]
        beta  = self.beta_proj(cond)       # [B, C]
        
        gamma = gamma[..., None, None]     # broadcast 到 H W
        beta  = beta[..., None, None]
        
        # 常见两种写法，任选
        # out = (1 + gamma) * out + beta           # FiLM-style LN，推荐（从identity开始）
        out = gamma * out + beta                   # 更接近原始 FiLM
        
        return out


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_init_value = layer_scale_init_value
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def init_weights(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""
    Code adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.pyConvNeXt

    A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf

    Args:
        size (str): Predefined model size. One of "tiny", "small", "base", "large". Default: "base".
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        patch_size (int | None): Pseudo patch size. Used to resize feature maps to those of a ViT with a given patch size. If None, no resizing is performed
        pretrained (str | None): Path to pretrained model. If None, no loading is performed.
    """

    def __init__(
        self,
        # original ConvNeXt arguments
        size: str = "base", # tiny / small / base / large
        layer_scale_init_value: float = 1e-6,
        # DINO arguments
        patch_size: int | None = None,
        pretrained: str | None = None,
    ):
        super().__init__()
        convnext_sizes = {
            "tiny": dict(depths=[3, 3, 9, 3],   dims=[96, 192, 384, 768]),
            "small": dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
            "base": dict(depths=[3, 3, 27, 3],  dims=[128, 256, 512, 1024]),
            "large": dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
            # 可选扩展：ConvNeXt-V2 或其他变体
            # "xlarge": dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]),
        }
        if size not in convnext_sizes:
            raise ValueError(
                f"Unsupported ConvNeXt size: {size}. "
                f"Supported: {list(convnext_sizes.keys())}"
            )

        # 自动获取对应的 depths 和 dims
        depths = convnext_sizes[size]["depths"]
        dims   = convnext_sizes[size]["dims"]
        # ==== ConvNeXt's original init =====
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # ==== End of ConvNeXt's original init =====

        # ==== DINO adaptation ====
        self.head = nn.Identity()  # remove classification head
        self.embed_dim = dims[-1]
        self.embed_dims = dims  # per layer dimensions
        self.n_blocks = len(self.downsample_layers)  # 4
        self.chunked_blocks = False
        self.n_storage_tokens = 0  # no registers

        self.norms = nn.ModuleList([nn.Identity() for i in range(3)])
        self.norms.append(self.norm)

        self.patch_size = patch_size
        self.input_pad_size = 4  # first convolution with kernel_size = 4, stride = 4

        if pretrained is not None:
            # Load pretrain model
            state_dict = torch.load(pretrained, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)

            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)
        for stage_id, stage in enumerate(self.stages):
            for block_id, block in enumerate(stage):
                if block.gamma is not None:
                    nn.init.constant_(self.stages[stage_id][block_id].gamma, block.layer_scale_init_value)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.inference_mode()
    def forward(self, x):
        ret = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            ret.append(x)
        return ret

    def get_embed_dims(self) -> List[int]:
        return self.embed_dims
