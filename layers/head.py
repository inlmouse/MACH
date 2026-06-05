import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.initialization import trunc_normal_
from layers.jepa import AuxMultimodalJEPABranch
from utils.detect_utils import make_anchors, dist2bbox
from layers.fusion import JointAttentionContrastiveHead

try:
    from flash_attn import flash_attn_varlen_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
    
class Prompt(nn.Module):
    """Prompt
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16, memory_blocks=128):
        super(Prompt, self).__init__()
        
        self.latent_len = num_feat // squeeze_factor
        self.subnet = nn.Sequential(
            nn.Linear(num_feat, self.latent_len , bias=False),
           )
        self.upnet= nn.Sequential(
            nn.Linear(self.latent_len , num_feat, bias=False),
        )
        self.mb =  torch.nn.Parameter(torch.randn(self.latent_len, memory_blocks))

    def forward(self, x):
        out1 = self.subnet(x)
        out1 = out1 @ self.mb
        out1 = F.softmax(out1, dim=-1)
        out = self.upnet( out1 @ self.mb.T )
        return out

class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.forward = self._forward

    def fuse(self):
        #del self.norm
        #del self.bias
        #del self.logit_scale
        self.forward = self._forward_fuse
    
    def unfuse(self):
        self.forward = self._forward

    def _forward_fuse(self, x, w):
        return x
    
    def _forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        # w = F.normalize(w, dim=-1, p=2)
        
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias

class TokenLevelBNContrastiveHead(nn.Module):
    """
    Token-level → Phrase-level 相似度头
    ------------------------------------------------
    输入
        x : (B, C, H, W)          # 图像特征（C == embed_dims）
        w : (B, P, L, C)          # token-level 文本特征
    输出
        sim_map   : (B, P, H, W)   # phrase 级像素相似度
        attn_map  : (B, P, L, H, W)  # （可选）每个 token 的注意力图
    ------------------------------------------------
    """

    def __init__(self, embed_dims: int, temperature_init: float = 0.07):
        super().__init__()
        self.embed_dims = embed_dims

        # 1. BN 替代 L2-norm（保持数值稳定）
        self.bn = nn.BatchNorm2d(embed_dims)

        # 2. 学习温度（等价于原来的 logit_scale）
        self.temperature = nn.Parameter(torch.full([], temperature_init))
        self.lse_temp = nn.Parameter(torch.tensor(5.0))

        # 4. 原始实现里的全局 bias（保持 loss 初始值一致）
        self.bias = nn.Parameter(torch.tensor([-10.0]))

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ):# -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x : (B, C, H, W)
        w : (B, P, L, C)   or  (B, P, C)  → 会自动 unsqueeze 为 (B,P,1,C)
        """
        B, C, H, W = x.shape
        if w.dim() == 3:                     # (B, P, C)
            w = w.unsqueeze(2)               # → (B, P, 1, C)
        B, P, L, C_ = w.shape
        assert C == C_, f"embed_dims mismatch: image {C} vs text {C_}"
        x = self.bn(x)                       # (B,C,H,W)
        sim = torch.einsum('bchw,bplc->bplhw', x, w)  # (B, P, L, H, W)
        sim = sim.contiguous().view(B * P, L, H * W)

        spatial_sim_pool = sim.mean(dim = -1)
        spatial_sim_pool = spatial_sim_pool.masked_fill(spatial_sim_pool == 0, -1e4)
        attn_weight = F.softmax(spatial_sim_pool, dim=-1)   # (B*P, L)

        # 加权求和
        phrase_map = (sim * attn_weight.unsqueeze(-1)).sum(dim=1)  # (B*P, H*W)
        phrase_map = phrase_map.view(B, P, H, W)                 # (B, P, H, W)
        
        phrase_map = phrase_map * self.temperature.exp() + self.bias

        return phrase_map
    
    def fuse(self, w: torch.Tensor):
        """
        将 w + BN + temperature + global_bias 融合成可复用的 conv 权重（batch-independent）。
        要求：
        - 推理时 w 只包含一个 "batch"（即全局共享），允许形状 (P, L, C) 或 (1, P, L, C)。
        - 在调用前请确保 self.bn 的 running_mean/var/weight/bias 已是训练后期望的值（通常在 eval 模式下调用）。
        结果：
        - 注册 buffer: _fused_weight (P*L, C, 1, 1), _fused_bias (P*L,), _P, _L, _fuse_mode(0)
        - 将 self.forward 指向 self.forward_fuse
        """
        # 允许 (P, L, C) 或 (1, P, L, C)
        if w.dim() == 3:
            w = w.unsqueeze(0)  # (P, L, C) -> (1, P, L, C)

        assert w.dim() == 4, "w must be (P, L, C) or (1, P, L, C)"
        assert w.size(0) == 1, "fuse expects w to have a single batch (1, P, L, C)"

        # 形状校验
        _, P, L, C = w.shape
        assert C == self.embed_dims, f"embed_dims mismatch: {C} vs {self.embed_dims}"

        # 使用 running stats（确保 eval 模式下调用以使用 running_mean/var）
        self.bn.eval()

        device = w.device
        dtype = w.dtype

        gamma = self.bn.weight.to(device=device, dtype=dtype).view(C)        # (C,)
        beta = self.bn.bias.to(device=device, dtype=dtype).view(C)           # (C,)
        mean = self.bn.running_mean.to(device=device, dtype=dtype).view(C)   # (C,)
        var = self.bn.running_var.to(device=device, dtype=dtype).view(C)     # (C,)
        eps = float(self.bn.eps)

        # 只取第 0 个 batch（因为推理时 w 在 batch 上是相同的）
        weight = w[0].contiguous().view(P * L, C)  # (PL, C)

        # per-channel scale 与 per-channel bias（BN 融合项）
        scale = gamma / torch.sqrt(var + eps)            # (C,)
        per_channel_bias = beta - scale * mean           # (C,)

        # 融合 weight（对每个滤波器的每个通道缩放）
        fused_weight = (weight * scale.view(1, C)).view(P * L, C, 1, 1)  # (PL, C,1,1)

        # 每个输出滤波器的 bias = sum_c (weight_jc * per_channel_bias_c)
        fused_bias = (weight * per_channel_bias.view(1, C)).sum(dim=1)   # (PL,)

        # 融合 temperature 与 global_bias （原 forward: sim = sim * temp + global_bias）
        #temp = self.temperature.exp().to(device=device, dtype=dtype)
        #fused_weight = fused_weight * temp
        #fused_bias = fused_bias * temp + self.bias.to(device=device, dtype=dtype)

        # 注册 buffer（与 batch 无关）
        # 注意：使用 register_buffer 可以确保它们随 model.to(...) 移动
        self.register_buffer('_fused_weight', fused_weight)   # (PL, C,1,1)
        self.register_buffer('_fused_bias', fused_bias)       # (PL,)
        self.register_buffer('_P', torch.tensor(P, device=device))
        self.register_buffer('_L', torch.tensor(L, device=device))

        # 替换 forward 为 fused 版本
        self.forward = self.forward_fuse


    def forward_fuse(self, x: torch.Tensor, w=None):
        """
        使用 fuse() 预融合后的权重进行推理。
        特点：
        - 不依赖 batch 维度（与 fuse 时的 batch 无关）
        - 忽略 w（已融合，无需输入）
        - 输出与原 forward 完全等价
        """
        assert hasattr(self, '_fused_weight'), "call fuse() before forward_fuse()"

        B, C, H, W = x.shape
        P = self._P.item()
        L = self._L.item()

        # fused_weight: (PL, C, 1, 1)
        # fused_bias:   (PL,)
        # conv2d 输出: (B, PL, H, W)
        token_maps = F.conv2d(
            x,
            self._fused_weight,
            bias=self._fused_bias,
            groups=1 
        )

        # reshape 回 (B, P, L, H, W)
        sim = token_maps.view(B, P, L, H, W)  # (B, P, L, H, W)
        #sim = sim * self.temperature.exp() + self.global_bias
        sim = sim.contiguous().view(B * P, L, H * W)

        spatial_sim_pool = sim.mean(dim = -1)
        spatial_sim_pool = spatial_sim_pool.masked_fill(spatial_sim_pool == 0, -1e4)
        # 用 LSE 产生注意力权重
        #lse = torch.logsumexp(sim / self.lse_temp, dim=-1) * self.lse_temp   # (B*P, L)
        attn_weight = F.softmax(spatial_sim_pool, dim=-1)   # (B*P, L)

        # 加权求和
        phrase_map = (sim * attn_weight.unsqueeze(-1)).sum(dim=1)  # (B*P, H*W)
        phrase_map = phrase_map.view(B, P, H, W)                 # (B, P, H, W)
        phrase_map = phrase_map * self.temperature.exp() + self.bias
        return phrase_map
    
class AttentionContrastiveHead(nn.Module):
    """
    语义对比头 (Attention Contrastive Head, MACH)
    支持双后端：
    1. PyTorch 原生 SDPA (Scaled Dot Product Attention)
    2. Flash Attention 2 (Varlen 模式)：针对带 Padding 的序列进行极致优化
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        use_flash_attn: bool = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = dim // 4
        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = self.hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 学习参数
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

        # 投影层
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim, affine=False)
        )
        self.kv_proj = nn.Linear(dim, self.hidden_dim * 2, bias=qkv_bias)
        self.score_proj = nn.Linear(self.hidden_dim, 1, bias=False)

        # 后端检测
        if use_flash_attn is None:
            use_flash_attn = _HAS_FLASH_ATTN
        if use_flash_attn and not _HAS_FLASH_ATTN:
            print("Warning: use_flash_attn=True but flash-attn not found. Falling back to SDPA.")
            use_flash_attn = False
        self.use_flash_attn = use_flash_attn

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, w: torch.Tensor, m: torch.Tensor = None):
        """
        Args:
            x: 图像特征 [B, Dim, H, W]
            w: 语义/文本特征 [B_nc, L, Dim] (B_nc = B * num_classes)
            m: Mask [B_nc, L], 1 为有效, 0 为 Padding
        """
        B, _, H, W = x.shape
        B_nc, L, _ = w.shape
        nc = B_nc // B
        M = H * W  # 图像像素总数 (Query 长度)

        # 1. Query 投影: [B, C, H, W] -> [B_nc, num_heads, M, head_dim]
        q = self.q_proj(x) 
        q = q.view(B, 1, self.hidden_dim, M).expand(B, nc, self.hidden_dim, M)
        q = q.reshape(B_nc, self.num_heads, self.head_dim, M).transpose(2, 3)

        # 2. KV 投影: [B_nc, L, Dim] -> [B_nc, num_heads, L, head_dim]
        kv = self.kv_proj(w).reshape(B_nc, L, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        # 3. 注意力计算
        if self.use_flash_attn and q.is_cuda:
            # Flash Attention 仅支持 CUDA 且非 FP32
            out = self._flash_attn_varlen(q, k, v, m)
        else:
            out = self._sdpa_attn(q, k, v, m)

        # 4. 后处理: [B_nc, num_heads, M, head_dim] -> [B, nc, H, W]
        out = out.transpose(1, 2).reshape(B_nc, M, self.hidden_dim)
        scores = self.score_proj(out).squeeze(-1)
        scores = scores.view(B, nc, H, W)
        # out = out.transpose(1, 2).reshape(B_nc, self.hidden_dim, H, W)
        
        # FiLM 风格的语义池化
        # scores = out.mean(dim=1).view(B, nc, H, W)
        return scores * self.logit_scale.exp() + self.bias

    def _sdpa_attn(self, q, k, v, m):
        """原生 SDPA 后端"""
        attn_mask = None
        if m is not None:
            # m: [B_nc, L] -> [B_nc, 1, 1, L]
            attn_mask = m.unsqueeze(1).unsqueeze(2).to(torch.bool)
        
        # 显式传递 scale 保证与 Flash Attention 一致
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
    
    def _flash_attn_varlen(self, q, k, v, m):
        """Flash Attention Varlen 后端：处理变长 KV 序列"""
        B_nc, h, M, d = q.shape
        L = k.shape[2]
        
        # Flash Attention 鲁棒性控制：强制转换 FP32 为 BF16/FP16
        orig_dtype = q.dtype
        if orig_dtype == torch.float32:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)

        # Q 侧：全部有效，直接展平 (Total_M, H, D)
        q_unpad = q.transpose(1, 2).reshape(-1, h, d).contiguous()
        cu_seqlens_q = torch.arange(0, B_nc + 1, device=q.device, dtype=torch.int32) * M

        # K/V 侧：根据 Mask 移除 Padding
        k_t = k.transpose(1, 2) # [B_nc, L, H, D]
        v_t = v.transpose(1, 2)
        
        if m is not None:
            # 找到非 Padding 的索引
            valid_idx = torch.nonzero(m.reshape(-1)).flatten()
            k_unpad = k_t.reshape(-1, h, d)[valid_idx].contiguous()
            v_unpad = v_t.reshape(-1, h, d)[valid_idx].contiguous()
            
            # 计算变长序列偏移
            seqlens_k = m.sum(dim=-1, dtype=torch.int32)
            cu_seqlens_k = torch.zeros(B_nc + 1, device=k.device, dtype=torch.int32)
            cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)
            max_s_k = seqlens_k.max().item()
        else:
            k_unpad = k_t.reshape(-1, h, d).contiguous()
            v_unpad = v_t.reshape(-1, h, d).contiguous()
            cu_seqlens_k = torch.arange(0, B_nc + 1, device=k.device, dtype=torch.int32) * L
            max_s_k = L

        # 调用核心函数
        out = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=M,
            max_seqlen_k=max_s_k,
            softmax_scale=self.scale,
            causal=False
        )

        # 还原维度并转换回原始精度
        out = out.view(B_nc, M, h, d).transpose(1, 2)
        return out.to(orig_dtype)

class ModulatedAttentionContrastiveHead(nn.Module):
    """
    语义对比头 (Modulated Attention Contrastive Head, MACH)
    支持双后端：
    1. PyTorch 原生 SDPA (Scaled Dot Product Attention)
    2. Flash Attention 2 (Varlen 模式)：针对带 Padding 的序列进行极致优化
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        use_flash_attn: bool = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = dim // 4
        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = self.hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 学习参数
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

        # 投影层
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim, affine=False)
        )
        self.kv_proj = nn.Linear(dim, self.hidden_dim * 2, bias=qkv_bias)
        self.text_to_gate = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim) # 输出 [B_nc, hidden_dim]
        )

        # 后端检测
        if use_flash_attn is None:
            use_flash_attn = _HAS_FLASH_ATTN
        if use_flash_attn and not _HAS_FLASH_ATTN:
            print("Warning: use_flash_attn=True but flash-attn not found. Falling back to SDPA.")
            use_flash_attn = False
        self.use_flash_attn = use_flash_attn

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, w: torch.Tensor, m: torch.Tensor = None):
        """
        Args:
            x: 图像特征 [B, Dim, H, W]
            w: 语义/文本特征 [B_nc, L, Dim] (B_nc = B * num_classes)
            m: Mask [B_nc, L], 1 为有效, 0 为 Padding
        """
        B, _, H, W = x.shape
        B_nc, L, _ = w.shape
        nc = B_nc // B
        M = H * W  # 图像像素总数 (Query 长度)

        # 1. Query 投影: [B, C, H, W] -> [B_nc, num_heads, M, head_dim]
        q = self.q_proj(x) 
        q = q.view(B, 1, self.hidden_dim, M).expand(B, nc, self.hidden_dim, M)
        q = q.reshape(B_nc, self.num_heads, self.head_dim, M).transpose(2, 3)

        # 2. KV 投影: [B_nc, L, Dim] -> [B_nc, num_heads, L, head_dim]
        kv = self.kv_proj(w).reshape(B_nc, L, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        # 3. 注意力计算
        if self.use_flash_attn and q.is_cuda:
            # Flash Attention 仅支持 CUDA 且非 FP32
            out = self._flash_attn_varlen(q, k, v, m)
        else:
            out = self._sdpa_attn(q, k, v, m)

        # 4. 后处理: [B_nc, num_heads, M, head_dim] -> [B, nc, H, W]
        out = out.transpose(1, 2).reshape(B_nc, M, self.hidden_dim)
        out = out.transpose(1, 2).reshape(B_nc, self.hidden_dim, H, W)
        
        # FiLM 风格的语义池化
        # scores = out.mean(dim=1).view(B, nc, H, W)
        # 1. 提取文本的全局摘要 (Global Text Summary) [B_nc, Dim]
        # w: [B_nc, L, Dim], m: [B_nc, L]
        text_mask = m.unsqueeze(-1).float() if m is not None else torch.ones_like(w[:, :, :1])
        w_summary = (w * text_mask).sum(dim=1) / (text_mask.sum(dim=1) + 1e-6)

        # 2. 动态预测当前 Query 专属的通道门控权重 -> reshape 为 [B_nc, hidden_dim, 1, 1]
        # 我们不加 Tanh 或 Sigmoid，允许网络输出负值，从而让文本能够主动“隐灭/抑制”无关的背景通道
        dynamic_weights = self.text_to_gate(w_summary).view(B_nc, self.hidden_dim, 1, 1)
        # 5. 【极限优化】利用广播机制进行动态 1x1 卷积与通道融合
        # 这一步没有任何卷积图的底层开销，只是纯粹的 element-wise mul 和 sum
        scores = (out * dynamic_weights).sum(dim=1) # [B_nc, H, W]
        scores = scores.view(B, nc, H, W)
        return scores * self.logit_scale.exp() + self.bias

    def _sdpa_attn(self, q, k, v, m):
        """原生 SDPA 后端"""
        attn_mask = None
        if m is not None:
            # m: [B_nc, L] -> [B_nc, 1, 1, L]
            attn_mask = m.unsqueeze(1).unsqueeze(2).to(torch.bool)
        
        # 显式传递 scale 保证与 Flash Attention 一致
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)

    def _flash_attn_varlen(self, q, k, v, m):
        """Flash Attention Varlen 后端：处理变长 KV 序列"""
        B_nc, h, M, d = q.shape
        L = k.shape[2]
        
        # Flash Attention 鲁棒性控制：强制转换 FP32 为 BF16/FP16
        orig_dtype = q.dtype
        if orig_dtype == torch.float32:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)

        # Q 侧：全部有效，直接展平 (Total_M, H, D)
        q_unpad = q.transpose(1, 2).reshape(-1, h, d).contiguous()
        cu_seqlens_q = torch.arange(0, B_nc + 1, device=q.device, dtype=torch.int32) * M

        # K/V 侧：根据 Mask 移除 Padding
        k_t = k.transpose(1, 2) # [B_nc, L, H, D]
        v_t = v.transpose(1, 2)
        
        if m is not None:
            # 找到非 Padding 的索引
            valid_idx = torch.nonzero(m.reshape(-1)).flatten()
            k_unpad = k_t.reshape(-1, h, d)[valid_idx].contiguous()
            v_unpad = v_t.reshape(-1, h, d)[valid_idx].contiguous()
            
            # 计算变长序列偏移
            seqlens_k = m.sum(dim=-1, dtype=torch.int32)
            cu_seqlens_k = torch.zeros(B_nc + 1, device=k.device, dtype=torch.int32)
            cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)
            max_s_k = seqlens_k.max().item()
        else:
            k_unpad = k_t.reshape(-1, h, d).contiguous()
            v_unpad = v_t.reshape(-1, h, d).contiguous()
            cu_seqlens_k = torch.arange(0, B_nc + 1, device=k.device, dtype=torch.int32) * L
            max_s_k = L

        # 调用核心函数
        out = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=M,
            max_seqlen_k=max_s_k,
            softmax_scale=self.scale,
            causal=False
        )

        # 还原维度并转换回原始精度
        out = out.view(B_nc, M, h, d).transpose(1, 2)
        return out.to(orig_dtype)

class ASMLH(nn.Module):
    """
    Token-level → Phrase-level 相似度头 (MACH 输入接口规范版)
    ------------------------------------------------
    输入:
        x : (B, C, H, W)          # 图像特征（C == embed_dims）
        w : (B_nc, L, C)          # Token-level 文本特征 (B_nc = B * P)
        m : (B_nc, L)             # Padding Mask, 1 为有效 Token, 0 为 Padding
    输出:
        phrase_map : (B, P, H, W) # Phrase 级像素置信度分数
    ------------------------------------------------
    """
    def __init__(self, embed_dims: int, temperature_init: float = 0.07):
        super().__init__()
        self.embed_dims = embed_dims

        # 1. BN 替代 L2-norm（保持数值稳定）
        self.bn = nn.BatchNorm2d(embed_dims)

        # 2. 学习温度与尺度参数
        self.temperature = nn.Parameter(torch.full([], temperature_init))
        # 3. 经典的先验绝对阈值偏置
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # 我们对内敛的 k 进行指数保护 k = exp(k_gate)，防止其变为负数导致逻辑反转。
        # 初始化为 0.0，意味着初始时 k = e^0 = 1.0，恰好处于 Mean(OR) 与 Min(AND) 的正中央。
        self.k_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        m: torch.Tensor = None
    ) -> torch.Tensor:
        
        B, C, H, W = x.shape
        
        # 🚀 降维保底：如果输入的 w 是 2D (B_nc, C)，自动补齐为 (B_nc, 1, C) 即 L=1
        if w.dim() == 2:
            w = w.unsqueeze(1)
            
        B_nc, L, C_ = w.shape
        assert C == C_, f"embed_dims mismatch: image {C} vs text {C_}"
        
        # 反推当前 Batch 中的 Phrase/Class 数量 P
        P = B_nc // B 

        # 1. 图像特征过 BN 稳定数值流形
        x = self.bn(x)  # (B, C, H, W)

        # 2. 恢复出临时 4D 结构，完美复用高效率的 einsum 矩阵乘法
        w_view = w.view(B, P, L, C)
        sim = torch.einsum('bchw,bplc->bplhw', x, w_view)  # (B, P, L, H, W)
        
        # 3. 展平空间维度，进入 MACH 经典的密集矩阵格式
        sim = sim.contiguous().view(B_nc, L, H * W)  # (B_nc, L, H*W)

        # 4. 空间全局池化：计算当前 Token 在全图上的平均语义唤醒度
        # spatial_sim_pool = sim.mean(dim=-1)  # (B_nc, L)
        # 动态释放动态严厉系数 k，增加 clamp 防止极端大值引发 FP16 溢出
        k = torch.exp(self.k_gate).clamp(min=1e-3, max=25.0)
        # 构建指数项：-k * X
        exponent = -k * sim  # [B_nc, L, M]
        if m is not None:
            # 完美的 1D 掩码广播：将 Padding 位置的指数项强行设为负无穷
            # 这样在接下来的 LogSumExp(e^-k*X) 连加中，Padding 位置对应的项变为 e^(-inf) = 0，被彻底净化
            txt_m = m.to(torch.bool).unsqueeze(-1)  # [B_nc, L, 1]
            exponent.masked_fill_(~txt_m, float('-inf'))
            
            # 计算非 Padding 的真实有效单词长度
            valid_lens = m.sum(dim=-1).view(B_nc, 1) # [B_nc, 1]
            valid_lens = torch.clamp(valid_lens, min=1.0)
            log_n = torch.log(valid_lens) # [B_nc, 1]
        else:
            log_n = torch.tensor(float(L), device=x.device).log()

        # 在 Token 维度（dim=1）执行高度优化的 Fused LogSumExp 算子
        # lse 形状: [B_nc, M]
        lse = torch.logsumexp(exponent, dim=1)

        # 补齐规范化项，并逆向除以 -k，将流形还原回相似度量纲
        # Soft-MIN = -1/k * (LogSumExp(-k*X) - log(N))
        phrase_map_flatten = -1.0 / k * (lse - log_n) # [B_nc, M]

        phrase_map = phrase_map_flatten.view(B, P, H, W)  # (B, P, H, W)
        
        phrase_map = phrase_map * self.temperature.exp() + self.bias

        return phrase_map       

class GMACH(nn.Module):
    """
    Geometric-Mean Attention Contrastive Head (G-MACH)
    
    核心数学特性：
    1. Sigmoid 独立激活：打破 Softmax 零和博弈，支持复杂组合属性的多重寻址。
    2. 零参数融合 (Zero-V)：直接抛弃 Value 矩阵，消除特征叠加带来的模长爆炸。
    3. Log-Sum-Mean (几何平均) 聚合：免疫长句稀释，赋予否定句式 "一票否决权"。
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qk_bias: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = dim // 4
        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = self.hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # SigLIP 风格的绝对阈值偏置 (极其重要！)
        # 初始化为较大的负数，防止初期海量负样本的 Sigmoid 激活淹没网络
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))

        # 仅保留 Q 和 K 的投影层，彻底抛弃 V 投影
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim, affine=False)
        )
        self.k_proj = nn.Linear(dim, self.hidden_dim, bias=qk_bias)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, w: torch.Tensor, m: torch.Tensor = None):
        """
        Args:
            x: 图像视觉特征 [B, Dim, H, W]
            w: 文本词向量特征 [B_nc, L, Dim] (文本侧保持完全 Frozen)
            m: Padding Mask [B_nc, L], 1 为有效词, 0 为 Padding
        """
        B, _, H, W = x.shape
        B_nc, L, _ = w.shape
        nc = B_nc // B
        M = H * W  # 图像像素总数

        # 1. 视觉 Query 投影与重塑: -> [B_nc, num_heads, M, head_dim]
        q = self.q_proj(x) 
        q = q.view(B, 1, self.hidden_dim, M).expand(B, nc, self.hidden_dim, M)
        q = q.reshape(B_nc, self.num_heads, self.head_dim, M).transpose(2, 3)

        # 2. 文本 Key 投影与重塑: -> [B_nc, num_heads, L, head_dim]
        k = self.k_proj(w).reshape(B_nc, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算 Raw Logits (内积)
        # q: [..., M, head_dim] x k.T: [..., head_dim, L] -> [B_nc, num_heads, M, L]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))  # [B_nc, num_heads, M, L]
        
        # 🚀 极致优化 1：全部改用原地(In-place)算子，零额外内存开销
        attn_logits.mul_(self.scale * self.logit_scale.exp()).add_(self.bias)

        # ==========================================
        # 4. 融合激活：彻底干掉中间的 probs 张量
        # ==========================================
        # 🚀 极致优化 2：使用 PyTorch 官方高度优化的 logsiogmoid 融合算子
        # 数学上等价于 torch.log(torch.sigmoid(x))，但速度极快且显存占用极小
        log_probs = F.logsigmoid(attn_logits) 
        
        # 腾出显存：手动解除对原始 logits 的引用
        del attn_logits

        # ==========================================
        # 5. Log-Sum-Mean 几何平均聚合
        # ==========================================
        if m is not None:
            # m: [B_nc, L] -> [B_nc, 1, 1, L]
            attn_mask = m.unsqueeze(1).unsqueeze(2).to(torch.bool)
            
            # 🚀 极致优化 3：使用原地掩码填充 
            log_probs.masked_fill_(~attn_mask, 0.0)
            
            # 🚀 致命 Bug 修复：将有效长度对齐为 3 维 [B_nc, 1, 1]！
            # 绝不能写成 4 维的 view(B_nc, 1, 1, 1)，否则会触发 B_nc^2 的恐怖广播
            valid_lens = m.sum(dim=-1).view(B_nc, 1, 1)
            valid_lens = torch.clamp(valid_lens, min=1.0)
            
            # 此时两边都是 3 维张量，PyTorch 会执行完美的、零开销的逐元素除法
            log_mean = log_probs.sum(dim=-1) / valid_lens
        else:
            log_mean = log_probs.mean(dim=-1)

        # 6. 还原回 (0, 1) 的概率打分空间
        # geom_scores 严格代表了该像素同时满足所有文本条件的 "综合几何得分"
        # geom_scores = torch.exp(log_mean) # [B_nc, num_heads, M]
        # 数学恒等转换：Logit = log_mean - log(1 - exp(log_mean))
        # 使用 torch.log1p(-torch.exp(log_mean)) 在 C++ 底层做到绝对数值稳定
        # 此时得到的 final_logits 完美脱离了 (0, 1) 限制，回到了全维 Logit 空间
        geom_scores = log_mean - torch.log1p(-torch.exp(log_mean) + 1e-7) # [B_nc, num_heads, M]

        # 7. 多头池化与空间还原
        # 多头捕获了不同的子空间语义，此处用均值聚合
        final_scores = geom_scores.mean(dim=1).view(B, nc, H, W)
        
        return final_scores

class Detect(nn.Module):
    """Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), text_embed_dim=0):
        """Initializes the detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.text_embed_dim = text_embed_dim  # 文本嵌入维度
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.tensor([8, 16, 32], dtype=torch.float32)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                # 第一层 3×3 卷积 (普通卷积，非 depthwise)
                nn.Conv2d(x, c2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(),
                
                # 第二层 3×3 卷积 (普通卷积，非 depthwise)
                nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(),
                
                # 最后一层 1×1 卷积 → 4 * reg_max（无 BN、无激活）
                nn.Conv2d(c2, 4 * self.reg_max, kernel_size=1, stride=1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                # 第一层 DWConv (depthwise 3x3)
                nn.Conv2d(x, x, kernel_size=3, stride=1, padding=1, groups=x, bias=False),
                nn.BatchNorm2d(x),
                nn.SiLU(),
                
                # 第二层 Pointwise 1x1 → c3
                nn.Conv2d(x, c3, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c3),
                nn.SiLU(),
                
                # 第三层 DWConv (depthwise 3x3)
                nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, groups=c3, bias=False),
                nn.BatchNorm2d(c3),
                nn.SiLU(),
                
                # 第四层 Pointwise 1x1 → c3
                nn.Conv2d(c3, c3, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c3),
                nn.SiLU(),
                
                # 最后一层 1x1 分类（无 BN、无激活）
                nn.Conv2d(c3, self.nc, kernel_size=1, stride=1) if self.text_embed_dim <= 0 else nn.Conv2d(c3, self.text_embed_dim, kernel_size=1, stride=1)
            )
            for x in ch
        )

        if self.text_embed_dim > 0:
            # self.cv3 = nn.ModuleList(nn.Sequential(nn.Conv2d(x, self.text_embed_dim, kernel_size=1, stride=1)) for x in ch)
            self.jepa = nn.ModuleList([AuxMultimodalJEPABranch(dim=text_embed_dim, reduction_ratio=2**(2-i)) for i in range(len(ch))])
            self.alignhead = nn.ModuleList(AttentionContrastiveHead(self.text_embed_dim, num_heads=3)  for _ in ch)
        # else:
        #     self.cv3 = nn.ModuleList(nn.Sequential(nn.Conv2d(x, self.nc, kernel_size=1, stride=1)) for x in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
           


    def forward(self, x, w = None, m=None, batch=None):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        jepa_loss = 0.0
        for i in range(self.nl):
            if self.text_embed_dim > 0 and w is not None:
                contras_feat = self.cv3[i](x[i])
                jepa_loss += self.jepa[i](contras_feat, w, m, batch)
                x[i] = torch.cat((self.cv2[i](x[i]), self.alignhead[i](contras_feat, w=w, m=m)), 1)
            else:
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        if self.training:  # Training path
            return x, jepa_loss
        y = self._inference(x)
        return (y, x)


    def _inference(self, x):
        """
        最简版推理前向：输入多尺度特征图列表 x，返回 (B, 4 + nc, num_anchors_total)
        其中 4 是 dbox (cxcywh)，nc 是类别概率（sigmoid 后）
        """
        # x: List[Tensor]，如 [P3, P4, P5]，每个是 BCHW
        shape = x[0].shape  # B, C, H, W
        dtype, device = x[0].dtype, x[0].device

        # 生成 anchors（如果还没有或形状不匹配）
        if not hasattr(self, 'anchors') or self.anchors is None or self.anchors.device != device:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        # 展平所有尺度 → (B, C, num_anchors_total)
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], dim=2)

        # 分离回归和分类部分
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # DFL → 软 argmax 得到分布中心 → 4 个坐标偏移
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0))  # (B, 4, num_total)

        # 乘以 stride 还原到原图尺度
        dbox = dbox * self.strides  # (B, 4, num_total)

        # 拼接 box + cls（cls 经过 sigmoid）
        return torch.cat((dbox, cls.sigmoid()), dim=1)  # (B, 4 + nc, num_total)

    def _save_original_cls_params(self):
        """保存原始分类头参数，用于支持多次调用 set_class"""
        if not hasattr(self, 'text_embed_dim') or self.text_embed_dim <= 0:
            return
        
        self._orig_cls_params = []
        for cls_head, bn_head in zip(self.cv3, self.alignhead):
            conv = cls_head[-1]  # 最后一层 1x1 卷积
            norm = bn_head.norm  # BatchNorm
            
            # 保存卷积权重和偏置
            params = {
                'weight': conv.weight.data.clone(),
                'bias': conv.bias.data.clone() if conv.bias is not None else None,
                'bn_weight': norm.weight.data.clone(),
                'bn_bias': norm.bias.data.clone(),
                'bn_running_mean': norm.running_mean.clone(),
                'bn_running_var': norm.running_var.clone(),
                'bn_eps': norm.eps,
                'in_channels': conv.in_channels,
                'out_channels': conv.out_channels,
            }
            self._orig_cls_params.append(params)
    
    def unset_class(self):
        """
        恢复 set_class 之前的原始参数状态。
        必须在 set_class 之后调用，且 _orig_cls_params 必须存在。
        """
        # 恢复原始类别数
        if hasattr(self, '_orig_nc'):
            self.nc = self._orig_nc
            self.no = self._orig_no
        return
        if not hasattr(self, '_orig_cls_params') or self._orig_cls_params is None:
            return  # 没有保存原始参数，无需恢复
        
        for i, (cls_head, bn_head) in enumerate(zip(self.cv3, self.alignhead)):
            if isinstance(bn_head, TokenLevelBNContrastiveHead):
                # TokenLevelBNContrastiveHead 的 fuse 是替换 forward 方法
                # 需要恢复原始 forward 方法
                if hasattr(bn_head, '_orig_forward'):
                    bn_head.forward = bn_head._orig_forward
                    delattr(bn_head, '_orig_forward')
                continue
            
            assert(isinstance(cls_head, nn.Sequential))
            assert(isinstance(bn_head, AttentionContrastiveHead))
            
            bn_head.unfuse()  # 恢复 BNContrastiveHead 的 forward 方法
            # 获取保存的原始参数
            orig_params = self._orig_cls_params[i]
            device = cls_head[-1].weight.device
            dtype = cls_head[-1].weight.dtype
            
            # 恢复卷积层到原始状态
            new_conv = nn.Conv2d(
                orig_params['in_channels'],
                orig_params['out_channels'],
                kernel_size=1,
            ).requires_grad_(True).to(device=device, dtype=dtype)
            
            new_conv.weight.data.copy_(orig_params['weight'])
            if orig_params['bias'] is not None:
                new_conv.bias.data.copy_(orig_params['bias'])
            
            cls_head[-1] = new_conv
            
            # 恢复 BN 层参数
            norm = bn_head.norm
            norm.weight.data.copy_(orig_params['bn_weight'])
            norm.bias.data.copy_(orig_params['bn_bias'])
            norm.running_mean.data.copy_(orig_params['bn_running_mean'])
            norm.running_var.data.copy_(orig_params['bn_running_var'])
    
    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, c, s in zip(m.cv2, m.cv3, m.alignhead, m.stride):  # from
            a[-1].bias.data[:] =2.0  # box
            # b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  


    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)

    def set_class(self, txt_feats):
        """
        动态设置类别文本特征，支持多次调用。
        
        Args:
            txt_feats: [num_classes, text_embed_dim] 文本特征张量
        """
        assert(not self.training)
        assert(self.text_embed_dim == txt_feats.shape[-1]), f"文本特征维度 {txt_feats.shape[-1]} 与 head 的 text_embed_dim {self.text_embed_dim} 不匹配"
        # 保存原始类别数和输出通道数
        if not hasattr(self, '_orig_nc'):
            self._orig_nc = self.nc
            self._orig_no = self.no
        self.nc = txt_feats.shape[0]  # 更新类别数
        self.no = self.nc + self.reg_max * 4  # 更新输出通道数
        return
        # 每次调用都保存当前参数（支持多次调用 set_class，始终基于当前状态）
        self._save_original_cls_params()
        
        #assert(hasattr(self, '_orig_cls_params')), "请先调用 _save_original_cls_params() 保存原始参数"
        
        # 获取 txt_feats 的设备和数据类型，确保所有计算在同一设备和类型上
        device = txt_feats.device
        dtype = txt_feats.dtype
        
        for i, (cls_head, bn_head) in enumerate(zip(self.cv3, self.alignhead)):
            if isinstance(bn_head, TokenLevelBNContrastiveHead):
                bn_head.fuse(txt_feats)
                continue
            assert(isinstance(cls_head, nn.Sequential))
            assert(isinstance(bn_head, BNContrastiveHead))
            
            # 使用保存的原始参数，而不是当前可能被修改过的 conv
            orig_params = self._orig_cls_params[i]
            
            logit_scale = bn_head.logit_scale.to(dtype=dtype)
            bias = bn_head.bias.to(dtype=dtype)
            
            # 从保存的参数中恢复卷积和 BN 权重，并转移到目标设备和类型
            w_conv = orig_params['weight'].to(device=device, dtype=dtype)  # [out_channels, in_channels, 1, 1]
            b_conv = orig_params['bias'].to(device=device, dtype=dtype) if orig_params['bias'] is not None else torch.zeros(orig_params['out_channels'], device=device, dtype=dtype)
            
            # 手动融合 conv 和 bn
            # BN: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
            bn_weight = orig_params['bn_weight'].to(device=device, dtype=dtype)
            bn_bias = orig_params['bn_bias'].to(device=device, dtype=dtype)
            bn_mean = orig_params['bn_running_mean'].to(device=device, dtype=dtype)
            bn_var = orig_params['bn_running_var'].to(device=device, dtype=dtype)
            bn_eps = orig_params['bn_eps']
            
            # 融合后的权重和偏置
            scale = bn_weight / torch.sqrt(bn_var + bn_eps)
            w_fused = w_conv * scale.view(-1, 1, 1, 1)  # [out_channels, in_channels, 1, 1]
            b_fused = (b_conv - bn_mean) * scale + bn_bias  # [out_channels]
            
            # 应用文本特征变换
            # w_fused: [text_embed_dim, in_channels, 1, 1] -> squeeze -> [text_embed_dim, in_channels]
            w = w_fused.squeeze(-1).squeeze(-1)  # [text_embed_dim, in_channels]
            b = b_fused  # [text_embed_dim]
            
            t = txt_feats * logit_scale.exp()  # [num_classes, text_embed_dim]
            
            # 新的权重: [num_classes, in_channels]
            w_new = t @ w
            # 新的偏置: [num_classes]
            b1 = (t @ b.unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias
            
            # 创建新的卷积层
            new_conv = nn.Conv2d(
                orig_params['in_channels'],
                w_new.shape[0],  # num_classes
                kernel_size=1,
            ).requires_grad_(False).to(device=device, dtype=dtype)
            
            new_conv.weight.data.copy_(w_new.unsqueeze(-1).unsqueeze(-1))
            new_conv.bias.data.copy_(b1 + b2)
            cls_head[-1] = new_conv
            
            # 融合 BN head（多次调用无副作用，fuse() 内部只是替换 forward 方法）
            bn_head.fuse()