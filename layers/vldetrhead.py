"""Transformer modules."""

from __future__ import annotations
from typing import Any

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, uniform_, xavier_uniform_
from utils.detect_utils import xyxy2xywh, xywh2xyxy, trunc_normal_

try:
    from flash_attn import flash_attn_varlen_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

__all__ = (
    "AIFI",
    "MLP",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "LayerNorm2d",
    "MLPBlock",
    "MSDeformAttn",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
)

def _get_clones(module, n):
    """Create a list of cloned modules from the given module.

    Args:
        module (nn.Module): The module to be cloned.
        n (int): Number of clones to create.

    Returns:
        (nn.ModuleList): A ModuleList containing n clones of the input module.

    Examples:
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def inverse_sigmoid(x, eps=1e-5):
    """Calculate the inverse sigmoid function for a tensor.

    This function applies the inverse of the sigmoid function to a tensor, which is useful in various neural network
    operations, particularly in attention mechanisms and coordinate transformations.

    Args:
        x (torch.Tensor): Input tensor with values in range [0, 1].
        eps (float, optional): Small epsilon value to prevent numerical instability.

    Returns:
        (torch.Tensor): Tensor after applying the inverse sigmoid function.

    Examples:
        >>> x = torch.tensor([0.2, 0.5, 0.8])
        >>> inverse_sigmoid(x)
        tensor([-1.3863,  0.0000,  1.3863])
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value.

    This function calculates the bias initialization value based on a prior probability using the inverse sigmoid
    (logit)
    function. It's commonly used in object detection models to initialize classification layers with a specific positive
    prediction probability.

    Args:
        prior_prob (float, optional): Prior probability for bias initialization.

    Returns:
        (float): Bias initialization value calculated from the prior probability.

    Examples:
        >>> bias = bias_init_with_prob(0.01)
        >>> print(f"Bias initialization value: {bias:.4f}")
        Bias initialization value: -4.5951
    """
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def linear_init(module):
    """Initialize the weights and biases of a linear module.

    This function initializes the weights of a linear module using a uniform distribution within bounds calculated from
    the output dimension. If the module has a bias, it is also initialized.

    Args:
        module (nn.Module): Linear module to initialize.

    Examples:
        >>> import torch.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> linear_init(linear)
    """
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Implement multi-scale deformable attention in PyTorch.

    This function performs deformable attention across multiple feature map scales, allowing the model to attend to
    different spatial locations with learned offsets.

    Args:
        value (torch.Tensor): The value tensor with shape (bs, num_keys, num_heads, embed_dims).
        value_spatial_shapes (torch.Tensor): Spatial shapes of the value tensor with shape (num_levels, 2).
        sampling_locations (torch.Tensor): The sampling locations with shape (bs, num_queries, num_heads, num_levels,
            num_points, 2).
        attention_weights (torch.Tensor): The attention weights with shape (bs, num_queries, num_heads, num_levels,
            num_points).

    Returns:
        (torch.Tensor): The output tensor with shape (bs, num_queries, num_heads * embed_dims).

    References:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()

def get_cdn_group(
    batch: dict[str, Any],
    num_queries: int,
    pooled_text: torch.Tensor, # 修改点 1：把 class_embed 改为了动态的 pooled_text [B, nc, hidden_dim]
    num_dn: int = 100,
    cls_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
    training: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
    """
    修改为支持动态文本输入的 CDN 去噪组生成器。
    
    Args:
        pooled_text: 形状为 [B, nc, hidden_dim]，当前 batch 的池化文本特征。
        其他参数意义与原版一致。
    """
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None
    
    gt_groups = batch["gt_groups"]
    text_is_positive = batch["text_is_positive"] # 来自batch['text_is_positive']的 bool tensor [B, nc]，记载了其中哪些是正样本
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None

    # 修改点 2：动态获取当前的类别数（nc），用于控制加噪时的随机上限
    bs, nc, hidden_dim = pooled_text.shape

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    
    gt_cls = batch["cls"].flatten().long()  # (bs*num, ) -> 这个 cls 应该是对应文本在 nc 维度的索引
    gt_bbox = batch["bboxes"]  # bs*num, 4
    b_idx = batch["batch_idx"] # bs*num, 表示每个 GT 属于 batch 中的哪张图

    # Each group has positive and negative queries
    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, )
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )

    # Positive and negative mask
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        # Apply class label noise to half of the samples
        mask = torch.rand(dn_cls.shape, device=dn_cls.device) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        if len(idx) > 0:
            # 1. 拿到这些需要加噪的 Query 对应的 batch 索引
            b_indices = dn_b_idx[idx]
            
            # 2. 提取对应的负样本掩码: [len(idx), nc]
            # True 表示是负样本/干扰词，False 表示是正样本
            text_is_neg = ~text_is_positive[b_indices].to(dn_cls.device)
            
            # 3. 向量化随机采样 (Gumbel-max 技巧变体)
            # 生成 [len(idx), nc] 的均匀分布随机数
            rand_scores = torch.rand(text_is_neg.shape, device=dn_cls.device)
            
            # 把正样本位置的得分强行拉到负无穷，彻底排查它们被选中的可能
            rand_scores.masked_fill_(~text_is_neg, -float('inf'))
            
            # 取最大值所在的索引，完美实现负样本中的随机采样！
            new_label = rand_scores.argmax(dim=1)
            
            # 赋值新标签
            dn_cls[idx] = new_label.to(dn_cls.dtype)

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)
        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 2*num_group*bs*num, 4
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # inverse sigmoid

    num_dn = int(max_nums * 2 * num_group)  # total denoising queries
    
    # -------------------------------------------------------------------------
    # 核心修改点 4：从二维动态字典提取特征
    # 以前是 dn_cls_embed = class_embed[dn_cls]
    # 现在 pooled_text 有 Batch 维度，必须通过 batch_idx 和 cls_idx 共同 Gather 特征
    # -------------------------------------------------------------------------
    dn_cls_embed = pooled_text[dn_b_idx, dn_cls]  # 高级索引 (Fancy Indexing)，输出 [bs*num * 2 * num_group, hidden_dim]

    padding_cls = torch.zeros(bs, num_dn, hidden_dim, device=gt_cls.device, dtype=dn_cls_embed.dtype)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device, dtype=dn_bbox.dtype)

    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)

    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # Match query cannot see the reconstruct
    attn_mask[num_dn:, :num_dn] = True
    # Reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i] = True
            
    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }

    return (
        padding_cls.to(pooled_text.device),
        padding_bbox.to(pooled_text.device),
        attn_mask.to(pooled_text.device),
        dn_meta,
    )


class TransformerEncoderLayer(nn.Module):
    """A single layer of the transformer encoder.

    This class implements a standard transformer encoder layer with multi-head attention and feedforward network,
    supporting both pre-normalization and post-normalization configurations.

    Attributes:
        ma (nn.MultiheadAttention): Multi-head attention module.
        fc1 (nn.Linear): First linear layer in the feedforward network.
        fc2 (nn.Linear): Second linear layer in the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after feedforward network.
        dropout (nn.Dropout): Dropout layer for the feedforward network.
        dropout1 (nn.Dropout): Dropout layer after attention.
        dropout2 (nn.Dropout): Dropout layer after feedforward network.
        act (nn.Module): Activation function.
        normalize_before (bool): Whether to apply normalization before attention and feedforward.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """Initialize the TransformerEncoderLayer with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform forward pass with post-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform forward pass with pre-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward propagate the input through the encoder module.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after transformer encoder layer.
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """AIFI transformer layer for 2D data with positional embeddings.

    This class extends TransformerEncoderLayer to work with 2D feature maps by adding 2D sine-cosine positional
    embeddings and handling the spatial dimensions appropriately.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """Initialize the AIFI instance with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the AIFI transformer layer.

        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W].

        Returns:
            (torch.Tensor): Output tensor with shape [B, C, H, W].
        """
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:
        """Build 2D sine-cosine position embedding.

        Args:
            w (int): Width of the feature map.
            h (int): Height of the feature map.
            embed_dim (int): Embedding dimension.
            temperature (float): Temperature for the sine/cosine functions.

        Returns:
            (torch.Tensor): Position embedding with shape [1, h*w, embed_dim].
        """
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c: int, num_heads: int):
        """Initialize a self-attention mechanism using linear transformations and multi-head attention.

        Args:
            c (int): Input and output channel dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a transformer block to the input x and return the output.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after transformer layer.
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer block based on https://arxiv.org/abs/2010.11929.

    This class implements a complete transformer block with optional convolution layer for channel adjustment, learnable
    position embedding, and multiple transformer layers.

    Attributes:
        conv (Conv, optional): Convolution layer if input and output channels differ.
        linear (nn.Linear): Learnable position embedding.
        tr (nn.Sequential): Sequential container of transformer layers.
        c2 (int): Output channel dimension.
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """Initialize a Transformer module with position embedding and specified number of heads and layers.

        Args:
            c1 (int): Input channel dimension.
            c2 (int): Output channel dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = nn.Sequential(nn.Conv2d(c1, c2, kernel_size=1), nn.BatchNorm2d(c2), nn.SiLU())#Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagate the input through the transformer block.

        Args:
            x (torch.Tensor): Input tensor with shape [b, c1, h, w].

        Returns:
            (torch.Tensor): Output tensor with shape [b, c2, h, w].
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, h, w = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, h, w)


class MLPBlock(nn.Module):
    """A single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function.

        Args:
            embedding_dim (int): Input and output dimension.
            mlp_dim (int): Hidden dimension.
            act (type): Activation function class.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP block.
        """
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """A simple multi-layer perceptron (also called FFN).

    This class implements a configurable MLP with multiple linear layers, activation functions, and optional sigmoid
    output activation.

    Attributes:
        num_layers (int): Number of layers in the MLP.
        layers (nn.ModuleList): List of linear layers.
        sigmoid (bool): Whether to apply sigmoid to the output.
        act (nn.Module): Activation function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act=nn.ReLU,
        sigmoid: bool = False,
        residual: bool = False,
        out_norm: nn.Module = None,
    ):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            num_layers (int): Number of layers.
            act (type): Activation function class.
            sigmoid (bool): Whether to apply sigmoid to the output.
            residual (bool): Whether to use residual connections.
            out_norm (nn.Module, optional): Normalization layer for the output.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.sigmoid = sigmoid
        self.act = act()
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # whether to apply a normalization layer to the output
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the entire MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP.
        """
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        if getattr(self, "residual", False):
            x = x + orig_x
        x = getattr(self, "out_norm", nn.Identity())(x)
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(nn.Module):
    """2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    This class implements layer normalization for 2D feature maps, normalizing across the channel dimension while
    preserving spatial dimensions.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter.
        bias (nn.Parameter): Learnable bias parameter.
        eps (float): Small constant for numerical stability.

    References:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """Initialize LayerNorm2d with the given parameters.

        Args:
            num_channels (int): Number of channels in the input.
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass for 2D layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Normalized output tensor.
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    This module implements multiscale deformable attention that can attend to features at multiple scales with learnable
    sampling locations and attention weights.

    Attributes:
        im2col_step (int): Step size for im2col operations.
        d_model (int): Model dimension.
        n_levels (int): Number of feature levels.
        n_heads (int): Number of attention heads.
        n_points (int): Number of sampling points per attention head per feature level.
        sampling_offsets (nn.Linear): Linear layer for generating sampling offsets.
        attention_weights (nn.Linear): Linear layer for generating attention weights.
        value_proj (nn.Linear): Linear layer for projecting values.
        output_proj (nn.Linear): Linear layer for projecting output.

    References:
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        """Initialize MSDeformAttn with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_levels (int): Number of feature levels.
            n_heads (int): Number of attention heads.
            n_points (int): Number of sampling points per attention head per feature level.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform forward pass for multiscale deformable attention.

        Args:
            query (torch.Tensor): Query tensor with shape [bs, query_length, C].
            refer_bbox (torch.Tensor): Reference bounding boxes with shape [bs, query_length, n_levels, 2 or 4], range
                in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area.
            value (torch.Tensor): Value tensor with shape [bs, value_length, C].
            value_shapes (list): List with shape [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})].
            value_mask (torch.Tensor, optional): Mask tensor with shape [bs, value_length], True for padding elements,
                False for non-padding elements.

        Returns:
            (torch.Tensor): Output tensor with shape [bs, Length_{query}, C].

        References:
            https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    This class implements a single decoder layer with self-attention, cross-attention using multiscale deformable
    attention, and a feedforward network.

    Attributes:
        self_attn (nn.MultiheadAttention): Self-attention module.
        dropout1 (nn.Dropout): Dropout after self-attention.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        cross_attn (MSDeformAttn): Cross-attention module.
        dropout2 (nn.Dropout): Dropout after cross-attention.
        norm2 (nn.LayerNorm): Layer normalization after cross-attention.
        linear1 (nn.Linear): First linear layer in the feedforward network.
        act (nn.Module): Activation function.
        dropout3 (nn.Dropout): Dropout in the feedforward network.
        linear2 (nn.Linear): Second linear layer in the feedforward network.
        dropout4 (nn.Dropout): Dropout after the feedforward network.
        norm3 (nn.LayerNorm): Layer normalization after the feedforward network.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            d_ffn (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            n_levels (int): Number of feature levels.
            n_points (int): Number of sampling points.
        """
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        # self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # self.catext_norm = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the Feed-Forward Network part of the layer.

        Args:
            tgt (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after FFN.
        """
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        text_dict: tuple,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform the forward pass through the entire decoder layer.

        Args:
            embed (torch.Tensor): Input embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Feature maps.
            shapes (list): Feature shapes.
            padding_mask (torch.Tensor, optional): Padding mask.
            attn_mask (torch.Tensor, optional): Attention mask.
            query_pos (torch.Tensor, optional): Query position embeddings.

        Returns:
            (torch.Tensor): Output tensor after decoder layer.
        """
        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # w, m = text_dict
        # B, nc, L, Dim = w.shape
        # w_flat = w.view(B, -1, Dim) # [B, nc * L, Dim]
        # m_flat = m.view(B, -1) # [B, nc * L]
        # key_padding_mask = ~m_flat.bool()
        
        # embed2 = self.ca_text(
        #     query=self.with_pos_embed(embed, query_pos).transpose(0, 1),
        #     key=w_flat.transpose(0, 1),
        #     value=w_flat.transpose(0, 1),
        #     key_padding_mask=key_padding_mask,
        #     attn_mask=None,
        #     need_weights=False  # 训练时设为 False 节省大量显存
        # )[0].transpose(0, 1)
        # embed = embed + self.catext_dropout(embed2)
        # embed = self.catext_norm(embed)

        # Cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """Deformable Transformer Decoder based on PaddleDetection implementation.

    This class implements a complete deformable transformer decoder with multiple decoder layers and prediction heads
    for bounding box regression and classification.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        num_layers (int): Number of decoder layers.
        hidden_dim (int): Hidden dimension.
        eval_idx (int): Index of the layer to use during evaluation.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        """Initialize the DeformableTransformerDecoder with the given parameters.

        Args:
            hidden_dim (int): Hidden dimension.
            decoder_layer (nn.Module): Decoder layer module.
            num_layers (int): Number of decoder layers.
            eval_idx (int): Index of the layer to use during evaluation.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed: torch.Tensor,  # decoder embeddings
        refer_bbox: torch.Tensor,  # anchor
        feats: torch.Tensor,  # image features
        shapes: list,  # feature shapes
        bbox_head: nn.Module,
        score_head: nn.Module,
        text_dict: tuple,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """Perform the forward pass through the entire decoder.

        Args:
            embed (torch.Tensor): Decoder embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Image features.
            shapes (list): Feature shapes.
            bbox_head (nn.Module): Bounding box prediction head.
            score_head (nn.Module): Score prediction head.
            text_dict (tuple): tuple contains unpooled text featture and mask
            pos_mlp (nn.Module): Position MLP.
            attn_mask (torch.Tensor, optional): Attention mask.
            padding_mask (torch.Tensor, optional): Padding mask.

        Returns:
            dec_bboxes (torch.Tensor): Decoded bounding boxes.
            dec_cls (torch.Tensor): Decoded classification scores.
        """
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        w, m = text_dict
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, text_dict, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output, w, m))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output, w, m))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
    

class EncoderContrastiveHead(nn.Module):
    """
    Encoder 专用的轻量化点积对比头
    通过图像特征与池化后的文本特征计算余弦相似度。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # 为了保证不同模态映射到统一空间，可以用一个线性层对特征做微调（可选）
        self.img_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 对比学习的温度系数，用于拉大分数差距，方便 top-k 选取
        # 初始化为 ln(10) ≈ 2.302，使得 exp(2.302) ≈ 10.0
        # 这样余弦相似度 [-1, 1] 会被拉伸到 [-10, 10]，足以对抗 Bias
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(10.0))
        # Focal Loss 先验概率初始化
        initial_bias = bias_init_with_prob(0.01)
        self.bias = nn.Parameter(torch.ones([]) * initial_bias)
        
        # 初始化权重 (防止初始特征随机爆炸)
        nn.init.xavier_uniform_(self.img_proj.weight)
        nn.init.constant_(self.img_proj.bias, 0)

    def forward(self, features: torch.Tensor, pooled_text: torch.Tensor):
        """
        Args:
            features: [B, M, hidden_dim] (M 为 8400 左右的多尺度拉平像素点)
            pooled_text: [B, nc, hidden_dim] (文本经过 Masked Mean Pooling 或 CLS 提取的特征)
        Returns:
            scores: [B, M, nc]
        """
        # 1. 图像特征投影并归一化
        feat_proj = self.img_proj(features)
        feat_norm = F.normalize(feat_proj, dim=-1) # [B, M, hidden_dim]
        
        # 2. 文本特征归一化
        text_norm = F.normalize(pooled_text, dim=-1) # [B, nc, hidden_dim]
        
        # 3. 计算余弦相似度 (点积)
        # b: batch, m: M 个像素点, n: nc 个类别/短语, d: hidden_dim
        sim = torch.einsum('bmd, bnd -> bmn', feat_norm, text_norm)
        
        # 4. 乘上温度系数放大分数差异
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        scores = sim * self.logit_scale.exp() + self.bias
        
        return scores


class AttentionContrastiveHead(nn.Module):
    """
    语义动态注意力对比头 (Attention Contrastive Head)
    用于 RT-DETR Decoder 的尾部分类阶段。
    
    输入: 
        - Queries [B, N, Dim] (如 N=300)
        - Text Features [B, nc, L, Dim] (未池化的文本词向量)
    输出: 
        - Scores [B, N, nc] (与原版 RT-DETR 分类输出维度完全一致)
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

        # 投影层：注意这里处理的是 1D 序列 [B, N, Dim]，所以使用 Linear 和 LayerNorm
        self.q_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.kv_proj = nn.Linear(dim, self.hidden_dim * 2, bias=qkv_bias)
        self.score_proj = nn.Linear(self.hidden_dim, 1)

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
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Focal Loss 先验概率初始化
        initial_bias = bias_init_with_prob(0.01)
        # 强制替换 score_proj 的初始偏置
        if hasattr(self, 'score_proj') and self.score_proj.bias is not None:
            nn.init.constant_(self.score_proj.bias, initial_bias)

    def _sdpa_attn(self, q, k, v, m):
        """
        原生的 PyTorch Scaled Dot-Product Attention 实现
        q: [B_nc, num_heads, N, head_dim]
        k, v: [B_nc, num_heads, L, head_dim]
        m: [B_nc, L] (1 表示有效 token, 0 表示 padding)
        """
        attn_mask = None
        if m is not None:
            # 必须把 mask 扩维到与 attention 矩阵 [B_nc, num_heads, N, L] 匹配
            # 形状变为 [B_nc, 1, 1, L]，PyTorch >= 2.0 中的 sdpa 支持 bool 型 mask
            # bool 型 mask 中 True 代表可以参与 Attention，False 代表 Padding。
            attn_mask = m.view(m.shape[0], 1, 1, m.shape[1]).bool()
            
            # 若使用 PyTorch < 2.0，则需要手动构造 float 掩码：
            # float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            # float_mask.masked_fill_(~attn_mask, float("-inf"))
            # attn_mask = float_mask
            
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return out

    def _flash_attn_varlen(self, q, k, v, m):
        """Flash Attention Varlen 后端：处理变长 KV 序列"""
        # 注意外部导入：from flash_attn import flash_attn_varlen_func
        
        B_nc, h, M, d = q.shape
        L = k.shape[2]
        
        # Flash Attention 鲁棒性控制：强制转换 FP32 为 BF16/FP16
        orig_dtype = q.dtype
        if orig_dtype == torch.float32:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)

        # 1. Q 侧：全部有效，直接展平 (B_nc * M, H, D)
        q_unpad = q.transpose(1, 2).reshape(-1, h, d).contiguous()
        cu_seqlens_q = torch.arange(0, B_nc + 1, device=q.device, dtype=torch.int32) * M

        # 2. K/V 侧：根据 Mask 移除 Padding
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
            
            # 🛡️ 护城河：防止全 Mask 导致 max_seqlen_k=0，引发底层 CUDA C++ 崩溃
            max_s_k = max(1, seqlens_k.max().item())
        else:
            k_unpad = k_t.reshape(-1, h, d).contiguous()
            v_unpad = v_t.reshape(-1, h, d).contiguous()
            cu_seqlens_k = torch.arange(0, B_nc + 1, device=k.device, dtype=torch.int32) * L
            max_s_k = L

        # 3. 计算 Softmax 缩放因子
        softmax_scale = d ** -0.5

        # 4. 调用核心函数
        out = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=M,
            max_seqlen_k=max_s_k,
            softmax_scale=softmax_scale,  # 修复了未定义的 self.scale
            causal=False
        )

        # 5. 还原维度并转换回原始精度
        out = out.view(B_nc, M, h, d).transpose(1, 2)
        return out.to(orig_dtype)

    def forward(self, x: torch.Tensor, w: torch.Tensor, m: torch.Tensor = None):
        """
        Args:
            x: Decoder 输出的 Object Queries [B, N, Dim] 
            w: 文本特征 [B, nc, L, Dim]  (nc = num_classes / 正负样本短语数)
            m: 文本 Mask [B, nc, L], 1 为有效 Token, 0 为 Padding Token
        Returns:
            scores: [B, N, nc]
        """
        B, N, Dim = x.shape
        _, nc, L, _ = w.shape
        
        B_nc = B * nc

        # 1. Query 投影: [B, N, Dim] -> [B, N, hidden_dim]
        q = self.q_proj(x)
        
        # 将 Query 复制 nc 份，为每个物体类别单独计算注意力
        # [B, N, hidden_dim] -> [B, 1, N, hidden_dim] -> [B, nc, N, hidden_dim]
        q = q.unsqueeze(1).expand(B, nc, N, self.hidden_dim)
        
        # Reshape 为 Attention 形状: [B_nc, num_heads, N, head_dim]
        # 注意：先 reshape 成 [B_nc, N, num_heads, head_dim]，再 transpose(1,2)
        q = q.reshape(B_nc, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. KV 投影: [B, nc, L, Dim] -> [B_nc, L, Dim * 2] -> ...
        kv = self.kv_proj(w).reshape(B_nc, L, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2) # 解绑后，k 和 v 的形状为 [B_nc, L, num_heads, head_dim]
        k, v = k.transpose(1, 2), v.transpose(1, 2) # 转置后 [B_nc, num_heads, L, head_dim]

        # 处理 Mask 对齐 B_nc
        if m is not None:
            m = m.reshape(B_nc, L)

        # 3. 注意力计算
        if self.use_flash_attn and q.is_cuda and q.dtype in [torch.float16, torch.bfloat16]:
            out = self._flash_attn_varlen(q, k, v, m)
        else:
            out = self._sdpa_attn(q, k, v, m)

        # 4. 后处理与打分
        # [B_nc, num_heads, N, head_dim] -> [B_nc, N, hidden_dim]
        out = out.transpose(1, 2).reshape(B_nc, N, self.hidden_dim)
        
        # 经过线性层打分: [B_nc, N, hidden_dim] -> [B_nc, N, 1] -> [B_nc, N]
        scores = self.score_proj(out).squeeze(-1)
        
        # 5. 还原回 DETR 标准的输出维度: [B_nc, N] -> [B, nc, N] -> [B, N, nc]
        scores = scores.view(B, nc, N).transpose(1, 2)
        
        # 加入缩放与偏置
        return scores


class VLRTDETRDecoder(nn.Module):
    """Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (EncoderContrastiveHead): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an VLRTDETRDecoder
        >>> decoder = VLRTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc #Full dynamic settings, no need to identify
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = EncoderContrastiveHead(hd)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([AttentionContrastiveHead(hd) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], w: torch.Tensor, m: torch.Tensor = None, batch: dict | None = None) -> tuple | torch.Tensor:
        """Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            w (torch.Tensor): 
            m (torch.Tensor): 
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, num_queries, 6) containing bounding boxes,
                confidence scores, and class labels.
        """

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)
        w = w.to(dtype=feats.dtype)
        if w.ndim == 3:
            _, L, C = w.shape
            w = w.reshape(feats.shape[0], -1, L, C)
            m = m.reshape(feats.shape[0], -1, L)
        pooled_text = self._get_last_token_pooling(w, m)
        _, nc, _ = pooled_text.shape
        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.num_queries,
            pooled_text, # [B, nc, hidden_dim]
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, pooled_text, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            (w, m),
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, num_queries, 4), (bs, num_queries, nc)
        y = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid(), nc)
        return y if self.export else (y, x)

    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor, nc: int) -> torch.Tensor:
        """Post-process predictions to select top-k detections.

        Args:
            boxes (torch.Tensor): Predicted bounding boxes with shape (batch_size, num_queries, 4) in xywh format.
            scores (torch.Tensor): Class scores with shape (batch_size, num_queries, nc).
            nc (int): Number of classes in one mini-batch

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, num_queries, 6) and last dimension format [cx,
                cy, w, h, max_class_prob, class_index].
        """
        scores, index = scores.flatten(1).topk(self.num_queries)
        query_idx = index // nc
        boxes = boxes.gather(dim=1, index=query_idx.unsqueeze(-1).expand(-1, -1, 4))
        return torch.cat([boxes, scores[..., None], (index % nc)[..., None].float()], dim=-1)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        pooled_text: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            pooled_text (torch.Tensor): Pooled text token from text encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # Prepare input for decoder
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features, pooled_text)  # (bs, h*w, nc)

        # Query selection
        # (bs*num_queries,)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs*num_queries,)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    @staticmethod
    def _get_last_token_pooling(w: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w: [B, nc, L, Dim] - 未池化的文本特征
            m: [B, nc, L]      - 文本 Mask (1 为有效，0 为 Padding)
        Returns:
            pooled_text: [B, nc, Dim] - 取出的最后一个有效 Token 的特征
        """
        B, nc, L, Dim = w.shape
        lengths = m.sum(dim=-1).long()
        last_idx = (lengths - 1).clamp(min=0)  # [B, nc]
        last_idx_expanded = last_idx.unsqueeze(-1).unsqueeze(-1).expand(B, nc, 1, Dim)
        pooled_text = w.gather(dim=2, index=last_idx_expanded).squeeze(2) # [B, nc, Dim]
        return pooled_text
    
    def _reset_parameters(self):
            """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
            # Class and bbox head init
            bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
            # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
            # linear_init(self.enc_score_head)
            # constant_(self.enc_score_head.bias, bias_cls)
            constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
            constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
            for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
                # linear_init(cls_)
                # constant_(cls_.score_proj.bias, bias_cls)
                constant_(reg_.layers[-1].weight, 0.0)
                constant_(reg_.layers[-1].bias, 0.0)

            linear_init(self.enc_output[0])
            xavier_uniform_(self.enc_output[0].weight)
            if self.learnt_init_query:
                xavier_uniform_(self.tgt_embed.weight)
            xavier_uniform_(self.query_pos_head.layers[0].weight)
            xavier_uniform_(self.query_pos_head.layers[1].weight)
            for layer in self.input_proj:
                xavier_uniform_(layer[0].weight)