import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.initialization import trunc_normal_
from layers.jepa import AuxMultimodalJEPABranch
from utils.detect_utils import make_anchors, dist2bbox

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

    def _forward_fuse(self, x, w, m=None):
        return x
    
    def _forward(self, x, w, m=None):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        # w = F.normalize(w, dim=-1, p=2)
        
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


    
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
        # self.score_proj = nn.Linear(self.hidden_dim, 1, bias=False)

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
        # scores = self.score_proj(out).squeeze(-1)
        # 语义池化打分（ach-woffn 版本：对 hidden_dim 取均值）
        scores = out.mean(dim=-1)
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
            self.alignhead = nn.ModuleList(AttentionContrastiveHead(self.text_embed_dim)  for _ in ch)
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
        if self.anchors is None or self.anchors.numel() == 0 or self.anchors.device != device:
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