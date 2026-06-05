import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.vitneck import RopePositionEmbedding, apply_rope

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    # if (mean < a - 2 * std) or (mean > b + 2 * std):
    #     warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std); u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1); tensor.erfinv_(); tensor.mul_(std * math.sqrt(2.)); tensor.add_(mean); tensor.clamp_(min=a, max=b)
        return tensor

class JointTransformerLayer(nn.Module):
    """
    轻量化瓶颈联合注意力层 (Low-Rank Joint Transformer Layer)
    在 dim // 4 空间计算注意力，降低 75% 的 Attention 算力消耗
    """
    def __init__(self, dim, num_heads=4, ffn_dim_mult=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 【核心优化】定义低维瓶颈空间
        self.hidden_dim = dim // 4
        assert self.hidden_dim % num_heads == 0, f"瓶颈维度 {self.hidden_dim} 必须能被 num_heads 整除"
        self.head_dim = self.hidden_dim // num_heads
        
        # Pre-LN 对应的 LayerNorm（仍在原生的满血 dim 维度）
        self.attn_norm_img = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn_norm_txt = nn.LayerNorm(dim, elementwise_affine=False)
        
        # 【核心优化】直接将高维投影到低维 [dim -> hidden_dim]
        self.q_proj = nn.Linear(dim, self.hidden_dim)
        self.k_proj = nn.Linear(dim, self.hidden_dim)
        self.v_proj = nn.Linear(dim, self.hidden_dim)
        
        # 输出投影负责将低维特征放回高维空间，以便进行满血的残差连接 [hidden_dim -> dim]
        self.out_proj = nn.Linear(self.hidden_dim, dim)
        
        # FFN 与对应的 Norm（保持原汁原味的高容量通道，不丧失网络深度表达力）
        self.ffn_norm_img = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn_norm_txt = nn.LayerNorm(dim, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_dim_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_dim_mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, img_seq: torch.Tensor, txt_seq: torch.Tensor, struct_mask: torch.Tensor, rope_sincos: tuple):
        """
        Args:
            img_seq: 纯图像内容流 [B, N_img, Dim]
            txt_seq: 文本特征流 [B, N_txt, Dim]
            struct_mask: 2D 联合类别隔离掩码 [B, 1, Total_Len, Total_Len]
            rope_sincos: 传入的 (sin, cos) 元组，用于图像 RoPE 旋转
        """
        B, N_img, _ = img_seq.shape
        _, N_txt, _ = txt_seq.shape
        Total_Len = N_img + N_txt
        
        # 1. 满血维度的 Pre-LN
        normed_img = self.attn_norm_img(img_seq)
        normed_txt = self.attn_norm_txt(txt_seq)
        
        # 2. 纯净内容流拼接 (作为 QKV 投影的基础输入)
        joint_input = torch.cat([normed_img, normed_txt], dim=1) # [B, Total_Len, dim]
        
        # 3. 【降维打击】直接投影至低维瓶颈空间，并切分多头
        # 投影后的形状: [B, num_heads, Total_Len, head_dim]
        q = self.q_proj(joint_input).view(B, Total_Len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(joint_input).view(B, Total_Len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(joint_input).view(B, Total_Len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 4. 【DETR 风格位置注入】在多头空间切出图像，精准施加 RoPE
        if rope_sincos is not None:
            sin, cos = rope_sincos # 形状应支持广播，如 [1, 1, N_img, head_dim]
            
            # 按照序列长度 N_img，把图像和文本的 Q、K 拆开
            q_img, q_txt = q[:, :, :N_img, :], q[:, :, N_img:, :]
            k_img, k_txt = k[:, :, :N_img, :], k[:, :, N_img:, :]
            
            # 仅对图像部分的多头向量进行位置旋转
            q_img = apply_rope(q_img, sin, cos)
            k_img = apply_rope(k_img, sin, cos)
            
            # 重新拼回完整的 Q 和 K 
            # (此时 Q, K 内部包含：带位置的图像 + 纯内容的文本；而 V 全程保持纯净内容，完美符合 DETR 范式)
            q = torch.cat([q_img, q_txt], dim=2)
            k = torch.cat([k_img, k_txt], dim=2)
        
        # 5. 低维空间的高效 Attention 计算
        context = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=struct_mask,
            dropout_p=0.1 if self.training else 0.0
        )
        
        # 6. 【升维回归】还原序列并放回原生满血维度
        context = context.permute(0, 2, 1, 3).flatten(2)       # [B, Total_Len, hidden_dim]
        attn_out = self.out_proj(context)                       # [B, Total_Len, dim]
        
        # 7. 切分回各模态，并走第一道残差
        img_seq = img_seq + attn_out[:, :N_img, :]
        txt_seq = txt_seq + attn_out[:, N_img:, :]
        
        # 8. 满血维度的独立 FFN 和第二道残差
        img_seq = img_seq + self.ffn(self.ffn_norm_img(img_seq))
        txt_seq = txt_seq + self.ffn(self.ffn_norm_txt(txt_seq))
        
        return img_seq, txt_seq


class JointTransformerEncoder(nn.Module):
    """
    负责多层串联堆叠的全局大网络
    """
    def __init__(self, dim, num_layers=6, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            JointTransformerLayer(dim, num_heads) for _ in range(num_layers)
        ])
        
        
    def forward(self, x: torch.Tensor, w: torch.Tensor, rope_sincos: torch.Tensor, m: torch.Tensor = None):
        B, Dim, H, W = x.shape
        B_nc, L, _ = w.shape
        num_classes = B_nc // B
        N_img = H * W
        N_txt = num_classes * L
        Total_Len = N_img + N_txt
        
        # 1. 仅在入口处执行一次：图像加位置编码并展平
        img_seq = x.flatten(2).permute(0, 2, 1)        # 纯图像特征流 [B, N_img, Dim]
        txt_seq = w.view(B, N_txt, Dim)
        
        # 2. 仅在入口处构造一次 2D 结构化掩码（全线复用，极其节省开销）
        struct_mask = torch.ones(Total_Len, Total_Len, dtype=torch.bool, device=x.device)
        txt_only_mask = torch.kron(
            torch.eye(num_classes, dtype=torch.bool, device=x.device),
            torch.ones(L, L, dtype=torch.bool, device=x.device)
        )
        struct_mask[N_img:, N_img:] = txt_only_mask
        # OVD任务这里禁止图像看文本！否则到了多层堆叠的下一层，文本 A 去读图像 Token 时，实际上间接读到了被文本 B 污染/影响过的视觉特征。（Semantic Washing）
        # struct_mask[:N_img, N_img:] = False
        
        if m is not None:
            txt_pad_mask = m.view(B, N_txt).to(torch.bool)
            img_pad_mask = torch.ones(B, N_img, dtype=torch.bool, device=x.device)
            col_pad_mask = torch.cat([img_pad_mask, txt_pad_mask], dim=1)
            final_mask = struct_mask.unsqueeze(0) & col_pad_mask.unsqueeze(1)
            attn_mask = final_mask.unsqueeze(1) # [B, 1, Total_Len, Total_Len]
        else:
            attn_mask = struct_mask.unsqueeze(0).unsqueeze(1)
            
        # 3. 核心：像接力棒一样，前一层的输出完美作为下一层的输入，不停精炼
        for layer in self.layers:
            img_seq, txt_seq = layer(img_seq, txt_seq, attn_mask, rope_sincos)
            
        # 4. 仅在出口处执行一次：将图像序列还原为 2D 空间特种形状
        updated_x = img_seq
        updated_w = txt_seq.view(B_nc, L, Dim)
        
        return updated_x, updated_w
    
class AttentionContrastiveModule(nn.Module):
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

        # 投影层
        self.q_proj = nn.Linear(dim, self.hidden_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, self.hidden_dim * 2, bias=qkv_bias)

        # 后端检测
        self.use_flash_attn = False

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

    def forward(self, x: torch.Tensor, w: torch.Tensor, H: int, W: int, m: torch.Tensor = None):
        """
        Args:
            x: 图像特征 [B, H*W, Dim]
            w: 语义/文本特征 [B_nc, L, Dim] (B_nc = B * num_classes)
            m: Mask [B_nc, L], 1 为有效, 0 为 Padding
        """
        B, M, _ = x.shape
        B_nc, L, _ = w.shape
        nc = B_nc // B
        # M = H * W  # 图像像素总数 (Query 长度)
        assert M == H * W

        # 1. Query 投影: [B, C, H, W] -> [B_nc, num_heads, M, head_dim]
        q = self.q_proj(x)  # [B, M, hidden_dim]
        q = q.unsqueeze(1).expand(B, nc, M, self.hidden_dim).reshape(B_nc, M, self.hidden_dim)
        q = q.reshape(B_nc, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B_nc, num_heads, M, head_dim]

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
        # scores = scores.view(B, nc, H, W)
        out = out.transpose(1, 2).reshape(B_nc, self.hidden_dim, H, W)
        
        # FiLM 风格的语义池化
        # scores = out.sum(dim=1).view(B, nc, H, W)
        scores = out.mean(dim=1).view(B, nc, H, W)
        return scores

    def _sdpa_attn(self, q, k, v, m):
        """原生 SDPA 后端"""
        attn_mask = None
        if m is not None:
            # m: [B_nc, L] -> [B_nc, 1, 1, L]
            attn_mask = m.unsqueeze(1).unsqueeze(2).to(torch.bool)
        
        # 显式传递 scale 保证与 Flash Attention 一致
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)  

class TokenLevelBNContrastiveHead(nn.Module):
    """
    Token-level → Phrase-level 相似度头 (1D 序列无缝对接版)
    ------------------------------------------------
    输入
        img_seq  : (B, M, C)           # 图像序列 (M = H * W, C == embed_dims)
        txt_seq  : (B, N_txt, C)       # 打平后的文本序列 (N_txt = nc * L)
    输出
        phrase_map : (B, nc, H, W)     # Phrase 级像素响应图
    ------------------------------------------------
    """
    def __init__(self, embed_dims: int):
        super().__init__()
        self.embed_dims = embed_dims

    def forward(
        self, 
        img_seq: torch.Tensor, 
        txt_seq: torch.Tensor, 
        H: int, 
        W: int, 
        m: torch.Tensor = None
    ):
        """
        Args:
            img_seq: Joint Transformer 输出的图像序列 [B, M, Dim] (M = H * W)
            txt_seq: Joint Transformer 输出的文本序列 [B, N_txt, Dim] (N_txt = nc * L)
            H, W: 原始图像的高和宽，仅用于最后一步生成响应图
            m: Mask [B_nc, L], 1 为有效, 0 为 Padding (B_nc = B * nc)
        """
        B, M, C = img_seq.shape
        
        # 1. 动态解析 nc (类别数) 和 L (每类最大 Token 长度)
        if m is not None:
            B_nc, L = m.shape
            nc = B_nc // B
        else:
            # 兜底逻辑：若完全没有传入 mask，默认每个短语只有 1 个 Token (L=1)
            L = 1
            nc = txt_seq.shape[1]
            B_nc = B * nc

        # 2. 恢复文本特征的多维结构: [B, nc * L, C] -> [B, nc, L, C]
        w = txt_seq.view(B, nc, L, C)

        # 3. 图像特征 1D BatchNorm 
        # img_seq: [B, M, C] -> transpose -> [B, C, M] -> BN -> transpose -> [B, M, C]
        # x = img_seq.transpose(1, 2)
        # x = self.bn(x).transpose(1, 2) # 此时数值与 2D BN 出来的完全一致
        x = F.normalize(img_seq, p=2, dim=-1) 
        w = F.normalize(w, p=2, dim=-1) 

        # 4. 计算 Token 级别的多模态点积相似度
        # x: [B, M, C], w: [B, nc, L, C] -> sim: [B, nc, L, M]
        sim = torch.einsum('bmc,bnlc->bnlm', x, w)
        
        # 打平 Batch 和 Class 维度，方便进行后续的空间池化
        sim = sim.reshape(B_nc, L, M)  # [B_nc, L, M]

        # 5. 空间池化：对像素维度 (M) 取平均，用于计算每个 Token 的激活权重
        spatial_sim_pool = sim.mean(dim=-1)  # [B_nc, L]

        # 6. 【精确封杀】使用真正的 Padding Mask 阻断无效 Token 的注意力分流
        if m is not None:
            # m 形状为 [B_nc, L]，1 为有效，0 为 Padding
            # 将 0 (Padding) 的地方强行填充为极小值，使其在之后的 Softmax 中权重归零
            spatial_sim_pool = spatial_sim_pool.masked_fill(~m.to(torch.bool), -1e4)
        else:
            # 原版代码的兜底过滤
            spatial_sim_pool = spatial_sim_pool.masked_fill(spatial_sim_pool == 0, -1e4)

        # 沿 Token 长度维度 (L) 计算 Softmax 权重
        attn_weight = F.softmax(spatial_sim_pool, dim=-1)   # [B_nc, L]

        # 7. Token 级局部特征 -> 加权求和融合为 Phrase 级全局特征
        # [B_nc, L, M] * [B_nc, L, 1] -> sum(dim=1) -> [B_nc, M]
        phrase_map = (sim * attn_weight.unsqueeze(-1)).sum(dim=1)
        
        # 8. 【全线唯一一次 2D 重组】将 Phrase 特征还原为标准的空间响应图
        phrase_map = phrase_map.view(B, nc, H, W)  # [B, nc, H, W]

        return phrase_map

class InnerProductSimilarity(nn.Module):
    """
    FILIP-style 极致轻量化延迟交互对齐头
    去除了所有多余的 QKV 投影层，纯粹利用内容流形空间的余弦相似度 + Max 聚合
    """
    def __init__(self):
        super().__init__()
        # alpha 控制平滑度：alpha 越大，越逼近纯 Max；alpha 越小，越逼近纯 Mean
        # 推荐设为 10.0 ~ 20.0 之间的较大值，使其成为一个平滑的“多值匹配 Max”
        self.alpha = nn.Parameter(torch.tensor([16.0]))

    def forward(
        self, 
        img_seq: torch.Tensor, 
        txt_seq: torch.Tensor, 
        m: torch.Tensor = None
    ):
        """
        Args:
            img_seq: [B, HW, D] —— 已经过 JointTransformer 充分淬炼的图像序列
            txt_seq: [B, nc * L, D] —— 已经过充分淬炼的打平文本序列
            H, W: 恢复响应图所需的空间分辨率
            m: Mask [B_nc, L], 1 为有效, 0 为 Padding
        """
        B, HW, D = img_seq.shape
        B_nc, L = m.shape if m is not None else (txt_seq.shape[0] * (txt_seq.shape[1] // txt_seq.shape[0]), txt_seq.shape[1])
        nc = B_nc // B

        # 1. 【极致稳定】L2 Normalize 归一化，将特征强行约束在单位超球面上
        # 彻底解决 Attention 机制中随着层数加深 Scale 容易飘、导致训练不稳定的顽疾
        img_node = F.normalize(img_seq, dim=-1)   # [B, HW, D]
        txt_node = F.normalize(txt_seq, dim=-1)   # [B, nc * L, D]

        # 2. 恢复文本特征的 Batch & Class 结构
        w = txt_node.view(B, nc, L, D)             # [B, nc, L, D]

        # 3. 跨模态 Token-level 相似度矩阵计算
        # 利用 einsum 规避掉频繁的 expand 内存开销
        # img_node: [B, HW, D], w: [B, nc, L, D] -> sim: [B, nc, HW, L]
        sim = torch.einsum('bmd,bnld->bnml', img_node, w)
        
        # 为了配合 Mask 动作，将 Batch 和 Class 维度合并
        sim = sim.reshape(B_nc, HW, L)             # [B_nc, HW, L]

        # 3. 【核心数学改变】使用 LogSumExp 实现平滑、连续的动态软对齐
        if m is not None:
            # 同样需要屏蔽 Padding：把 Padding 地方乘上平滑系数后，用极小值屏蔽
            # 确保 exp(-1e4) -> 0，对 sum 毫无贡献
            m_expanded = m.unsqueeze(1) # [B_nc, 1, L]
            sim_scaled = sim * self.alpha
            sim_scaled = sim_scaled.masked_fill(~m_expanded.to(torch.bool), -1e4)
            
            # 执行 LogSumExp
            score = torch.logsumexp(sim_scaled, dim=-1) / self.alpha
        else:
            score = torch.logsumexp(sim_scaled * self.alpha, dim=-1) / self.alpha # [B_nc, HW]

        return score

class JointAttentionContrastiveHead(nn.Module):
    def __init__(self, dim, num_heads=3, num_layers=6):
        super().__init__()
        
        self.rope_embed = RopePositionEmbedding(
            embed_dim=dim//4, num_heads=num_heads, base=100.0,
            normalize_coords="separate", shift_coords=None, jitter_coords=None,
            rescale_coords=None, dtype=None, device=None,
        )

        # 前面封装好的多层 Joint Transformer
        self.encoder = JointTransformerEncoder(dim, num_layers=num_layers, num_heads=num_heads)
        # 重构后的 1D 语义对比头
        # self.contrastive_head = AttentionContrastiveModule(dim, num_heads=num_heads)
        self.contrastive_head = InnerProductSimilarity()
        # 学习参数
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w, m=None):
        B, C, H, W = x.shape
        rope_sincos = self.rope_embed(H=H, W=W)
        # 1. 内部以 1D 形式进入，并经过 N 层不断精炼，出来时仍保持 1D 序列
        # img_seq: [B, H*W, Dim], txt_seq: [B, nc*L, Dim]
        img_seq, txt_seq = self.encoder(x, w, rope_sincos, m)
        
        # 2. 直接无缝无重组地丢入ACH 头
        # 只有在 MACH 的最后一行，才会触发唯一一次 view(B, nc, H, W) 
        response_map = self.contrastive_head(img_seq, txt_seq, m)
        response_map = response_map.view(B, -1, H, W)
        return response_map * self.logit_scale.exp() + self.bias