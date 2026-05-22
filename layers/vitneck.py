from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Literal, Tuple

try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        assert head_dim % 4 == 0, "Head dimension must be divisible by 4 for 2D RoPE"
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = head_dim
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(head_dim // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else: # min
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            coords += torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)[None, :]
        if self.training and self.jitter_coords is not None:
            jitter = (torch.empty(2, **dd).uniform_(-np.log(self.jitter_coords), np.log(self.jitter_coords))).exp()
            coords *= jitter[None, :]
        if self.training and self.rescale_coords is not None:
            rescale = (torch.empty(1, **dd).uniform_(-np.log(self.rescale_coords), np.log(self.rescale_coords))).exp()
            coords *= rescale

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).repeat(1, 2)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        if self.base is not None:
            periods = self.base ** (2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2))
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = self.max_period * (base ** (exponents - 1))
        self.periods.data.copy_(periods)

class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() #if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    # if (mean < a - 2 * std) or (mean > b + 2 * std):
    #     warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std); u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1); tensor.erfinv_(); tensor.mul_(std * math.sqrt(2.)); tensor.add_(mean); tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 use_flash_attn: bool = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 后端选择逻辑
        if use_flash_attn is None:
            use_flash_attn = _HAS_FLASH_ATTN
        if use_flash_attn and not _HAS_FLASH_ATTN:
            # 优雅降级而非崩溃（可选）
            import warnings
            warnings.warn("Flash Attention not installed, falling back to SDPA.")
            use_flash_attn = False
        self.use_flash_attn = use_flash_attn

    def forward(self, x, rope_sincos=None):
        B, N, C = x.shape
        
        # 1. QKV 投影 -> [B, N, 3, H, D] -> [B, N, H, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # 2. 处理 RoPE
        # 如果你的 apply_rope 支持 (B, N, H, D)，直接处理最快
        # 如果必须 (B, H, N, D)，则在这里转置一次
        if rope_sincos is not None:
            # 假设 apply_rope 需要 [B, H, N, D]
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            sin, cos = rope_sincos
            q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]
            q_patch = apply_rope(q_patch, sin, cos)
            k_patch = apply_rope(k_patch, sin, cos)
            q = torch.cat((q_cls, q_patch), dim=2)
            k = torch.cat((k_cls, k_patch), dim=2)
            
            # 注意：此时 q, k, v 形状为 [B, H, N, D]
        else:
            # 没 RoPE 时，保持 [B, N, H, D]
            pass

        # 3. Attention 分支
        if self.use_flash_attn and q.is_cuda:
            x = self._flash_attn(q, k, v) # 内部处理维度和精度
        else:
            x = self._sdpa_attn(q, k, v)

        # 4. 公共后处理
        # 确保输出回到 [B, N, C]
        # 如果 x 是 [B, H, N, D] -> transpose(1, 2)
        # 如果 x 是 [B, N, H, D] -> 不需要 transpose
        if x.shape[1] == self.num_heads: # 说明是 [B, H, N, D]
            x = x.transpose(1, 2)
            
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _sdpa_attn(self, q, k, v):
        """
        SDPA 支持多种维度，但 [B, H, N, D] 是最标准的。
        """
        if q.shape[1] != self.num_heads: # 如果还是 [B, N, H, D]
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            
        return F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_drop if self.training else 0.0,
            scale=self.scale # 显式对齐 scale
        )

    def _flash_attn(self, q, k, v):
        """
        Flash Attention 极致优化版
        """
        # 维度对齐：FA 强制要求 [B, N, H, D]
        if q.shape[1] == self.num_heads: # 如果是 [B, H, N, D]
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        
        # 精度检查：FA 不支持 FP32
        orig_dtype = q.dtype
        if orig_dtype == torch.float32:
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)
        
        # Flash Attention 核心调用
        # 注意：contiguous() 很重要，防止 transpose 后的内存不连续导致报错
        out = flash_attn_func(
            q.contiguous(), k.contiguous(), v.contiguous(), 
            dropout_p=self.attn_drop if self.training else 0.0, 
            softmax_scale=self.scale,
            causal=False
        )
        
        return out.to(orig_dtype) # 还原精度，保持后续网络一致性

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, w, m=None):
        """
        x: [B, M, C]      图像 token (Q 来源)
        w: [B, N, C]      文本 token (K/V 来源), N=L*nc
        m: [B, N]         padding mask, 1=有效, 0=pad
        """
        B, M, C = x.shape
        N = w.shape[1]

        # Q: [B, num_heads, M, head_dim]
        q = self.q_proj(x).reshape(B, M, self.num_heads, C // self.num_heads).transpose(1, 2)

        # K, V: [B, num_heads, N, head_dim] —— 不额外加位置编码
        kv = self.kv_proj(w).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        k, v = kv.unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        # Mask
        attn_mask = None
        if m is not None:
            attn_mask = m.unsqueeze(1).unsqueeze(2).bool()

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop
        )

        out = out.transpose(1, 2).reshape(B, M, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, sin, cos):
    """Applies RoPE to the input tensor."""
    return (x * cos) + (rotate_half(x) * sin)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.act(self.fc1(x)) 
        x = self.drop(x) 
        x = self.fc2(x) 
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.SiLU, norm_layer=nn.LayerNorm, ffn_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * ffn_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, rope_sincos=None):
        attn_output = self.attn(self.norm1(x), rope_sincos=rope_sincos)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class VisionTransformer(nn.Module):
    def __init__(
        self,  embed_dim=192, depth=6, num_heads=3, ffn_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., return_layers=[4, 5], norm_layer=None, act_layer=None, ffn_layer=Mlp
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.return_layers = return_layers
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.register_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer, ffn_layer=ffn_layer,
            ) for i in range(depth)
        ])

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim, num_heads=num_heads, base=100.0,
            normalize_coords="separate", shift_coords=None, jitter_coords=None,
            rescale_coords=None, dtype=None, device=None,
        )
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_vit_weights)
        self.rope_embed._init_weights()
        trunc_normal_(self.register_token, std=.02)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        outs = []
        x_embed = x
        _, _, H, W = x_embed.shape
        
        x_embed = x_embed.flatten(2).transpose(1, 2)
        register_token = self.register_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((register_token, x_embed), dim=1)
        rope_sincos = self.rope_embed(H=H, W=W)

        for i, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos=rope_sincos)
            if i in self.return_layers:
                outs.append(x[:, 1:])
        return outs


class ViTAdapter(nn.Module):
    
    def __init__(
        self,
        embed_dim=192,
        proj_dim=[],
        num_levels=3,
    ):
        super().__init__()
        self.num_levels = num_levels
        
        if num_levels != 3 and num_levels==len(proj_dim):
            raise NotImplementedError("Only support num_levels=3 for ViTAdapter now.")

        self.proj_dim = proj_dim

        self.projector = nn.ModuleList([ConvNormLayer_fuse(embed_dim, dim, kernel_size=1, stride=1) for dim in self.proj_dim])


    
    def forward(self, x, H_c = 40, W_c = 40):
        # fused_feats = (return_layers[0] + return_layers[1]) / 2
        fused_feats = torch.mean(torch.stack(x), dim=0)

        bs, _, _ = fused_feats.shape

        proj_feats = []
        fused_feats = fused_feats.transpose(1, 2).contiguous().view(bs, -1, H_c, W_c)  # [B, D, H, W]
        for i in range(self.num_levels):
            scale = 2 ** (1 - i)
            resize_H = int(H_c * scale)
            resize_W = int(W_c * scale)
            feature = F.interpolate(fused_feats, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            proj_feats.append(feature)
            
        if len(self.projector) == 1:
            proj_feats[-1] = self.projector[-1](proj_feats[-1])
        else:
            proj_feats = [layer(feat) for layer, feat in zip(self.projector, proj_feats)]
            
        return proj_feats

class ViTNeck(nn.Module):
    def __init__(self, embed_dim=192, proj_dim=[256, 512, 1024], depth=12, num_heads=3, ffn_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., return_layers=[10, 11], norm_layer=None, act_layer=None, ffn_layer=Mlp):
        super().__init__()
        assert len(proj_dim) == 3, "Currently only support 3 levels of features for ViTNeck."
        self.down_proj = nn.ModuleList([ConvNormLayer_fuse(proj_dim[0], embed_dim, kernel_size=1, stride=1), 
                                        ConvNormLayer_fuse(proj_dim[1], embed_dim, kernel_size=1, stride=1),
                                        ConvNormLayer_fuse(proj_dim[2], embed_dim, kernel_size=1, stride=1)])
        self.vit = VisionTransformer(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, return_layers=return_layers, norm_layer=norm_layer, act_layer=act_layer, ffn_layer=ffn_layer
        )
        self.vit_adapter = ViTAdapter(embed_dim=embed_dim, proj_dim=proj_dim, num_levels=len(proj_dim))
    
    def forward(self, x, w=None, batch=None):
        dino_feats = []
        _,_,H_c, W_c = x[1].shape
        for i in range(len(x)):
            x[i] = F.interpolate(x[i], size=[H_c, W_c], mode="bilinear", align_corners=False)
            x[i] = self.down_proj[i](x[i])
            dino_feats.append(x[i])

        x = torch.mean(torch.stack(dino_feats), dim=0)
        x = self.vit(x)
        return self.vit_adapter(x, H_c, W_c)

class MultiScaleViTFusion(nn.Module):
    def __init__(self, embed_dim=192, proj_dim=[256, 512, 1024], depth=2, num_heads=3, ffn_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., return_layers=[0, 1], norm_layer=None, act_layer=None, ffn_layer=Mlp):
        super().__init__()
        assert len(proj_dim) == 3, "Currently only support 3 levels of features for ViTFusion."
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.down_proj = nn.ModuleList([ConvNormLayer_fuse(proj_dim[i], embed_dim, kernel_size=1, stride=1) for i in range(len(proj_dim))])
        self.vit = nn.ModuleList([
            VisionTransformer(
                embed_dim=embed_dim, depth=depth, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, return_layers=return_layers, norm_layer=norm_layer, act_layer=act_layer, ffn_layer=ffn_layer
            ) for _ in range(len(proj_dim))
        ])
        self.up_proj = nn.ModuleList([ConvNormLayer_fuse(embed_dim, proj_dim[i], kernel_size=1, stride=1) for i in range(len(proj_dim))])
    
    def forward(self, x, w=None, batch=None):
        for i in range(len(x)):
            downproj_feat = self.down_proj[i](x[i]) #[N, embed_dim, H, W]
            vitfeats = self.vit[i](downproj_feat)   # List[Tensor]，Tensor shape: [N, H*W, embed_dim]
            avevitfeat = torch.mean(torch.stack(vitfeats), dim=0) # [N, H*W, embed_dim]
            _, _, H, W = x[i].shape
            downproj_feat = avevitfeat.transpose(1, 2).contiguous().view(x[i].shape[0], -1, H, W)  # [N, embed_dim, H, W]
            x[i] = self.up_proj[i](downproj_feat) #[N, proj_dim[i], H, W]
        return x # List[Tensor]，Tensor shape: [N, proj_dim[i], H, W]