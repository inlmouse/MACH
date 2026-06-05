import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 模块 1: 跨模态 SRA 预测器 (Unpooled Token 级融合)
# =====================================================================
class SRA_Predictor(nn.Module):
    """
    Unpooled SRA Cross-Modal Predictor (Bottleneck Attention Version)
    引入 hidden_dim 机制，大幅降低显存占用，并强化潜空间的信息瓶颈过滤能力。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        reduction_ratio: int = 2
    ):
        super().__init__()
        
        # ==========================================
        # 🛡️ 核心优化：定义低秩瓶颈维度
        # ==========================================
        self.hidden_dim = dim // 4
        
        assert self.hidden_dim % num_heads == 0, f"hidden_dim ({self.hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"

        self.dim = dim
        self.num_heads = num_heads
        # 注意：这里的 head_dim 是基于 hidden_dim 计算的
        self.head_dim = self.hidden_dim // num_heads

        # PreNorm：输入依然是原维度 dim
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm_vis = nn.LayerNorm(dim)
        self.kv_norm_txt = nn.LayerNorm(dim)

        # Query 降维投影：[dim -> hidden_dim]
        self.q_proj = nn.Linear(dim, self.hidden_dim)

        # 视觉空间缩减 (维持原维度处理空间，保留足够的感受野)
        self.sr = nn.Conv2d(
            dim,
            dim,
            kernel_size=reduction_ratio,
            stride=reduction_ratio,
            bias=False,
            groups=dim
        )

        # K/V 降维投影：[dim -> hidden_dim * 2] (包含 K 和 V)
        self.kv_vis_proj = nn.Linear(dim, self.hidden_dim * 2)
        self.kv_text_proj = nn.Linear(dim, self.hidden_dim * 2)

        # ==========================================
        # 🛡️ 核心优化：输出通道扩维解码
        # 从 hidden_dim 恢复到原空间 dim
        # ==========================================
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x_m, w_seq, text_mask):
        """
        x_m:       [N_valid, C, H, W] 掩码破坏后的视觉特征
        w_seq:     [N_valid, L, C] 对应正样本的文本 Token 序列
        text_mask: [N_valid, L] 文本的 PAD 掩码 (True 代表有效词)
        """
        N_valid, C, H, W = x_m.shape
        N = H * W

        # --------------------------------------------------
        # 1. 视觉 Query 生成 (降维到 hidden_dim)
        # --------------------------------------------------
        q_seq = x_m.flatten(2).transpose(1, 2)
        q_seq = self.q_norm(q_seq)
        
        q = self.q_proj(q_seq) # [N_valid, N, hidden_dim]
        
        q = q.reshape(
            N_valid, N, self.num_heads, self.head_dim
        ).transpose(1, 2) # [N_valid, heads, N, head_dim]

        # --------------------------------------------------
        # 2. 视觉 K/V 生成 (空间缩减 + 通道降维)
        # --------------------------------------------------
        x_reduced = self.sr(x_m)
        Nr = x_reduced.shape[2] * x_reduced.shape[3]
        
        x_r_seq = x_reduced.flatten(2).transpose(1, 2)
        x_r_seq = self.kv_norm_vis(x_r_seq)
        
        kv_vis = self.kv_vis_proj(x_r_seq) # [N_valid, Nr, hidden_dim * 2]
        
        kv_vis = kv_vis.reshape(
            N_valid, Nr, 2, self.num_heads, self.head_dim
        )
        k_vis, v_vis = kv_vis.unbind(2)
        k_vis, v_vis = k_vis.transpose(1, 2), v_vis.transpose(1, 2) # [N_valid, heads, Nr, head_dim]

        # --------------------------------------------------
        # 3. 文本 K/V 生成 (通道降维)
        # --------------------------------------------------
        w_seq = self.kv_norm_txt(w_seq)
        L = w_seq.shape[1]
        
        kv_txt = self.kv_text_proj(w_seq) # [N_valid, L, hidden_dim * 2]
        
        kv_txt = kv_txt.reshape(
            N_valid, L, 2, self.num_heads, self.head_dim
        )
        k_txt, v_txt = kv_txt.unbind(2)
        k_txt, v_txt = k_txt.transpose(1, 2), v_txt.transpose(1, 2) # [N_valid, heads, L, head_dim]

        # --------------------------------------------------
        # 4. 跨模态潜空间融合与 Attention 计算
        # --------------------------------------------------
        k = torch.cat([k_vis, k_txt], dim=2)
        v = torch.cat([v_vis, v_txt], dim=2)

        mask_vis = torch.ones(
            (N_valid, Nr), dtype=torch.bool, device=x_m.device
        )
        mask_kv = torch.cat([mask_vis, text_mask.bool()], dim=1)
        attn_mask = mask_kv[:, None, None, :]

        # PyTorch 2.0 原生极速 Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0
        ) # 输出形状: [N_valid, heads, N, head_dim]

        # --------------------------------------------------
        # 5. 通道扩维解码：从 hidden_dim 恢复到 C
        # --------------------------------------------------
        # 还原回 3D 序列形状
        out = out.transpose(1, 2).reshape(N_valid, N, self.hidden_dim)
        
        # 通过解码 MLP 恢复到主特征流形维度
        out = self.out_proj(out) # [N_valid, N, C]
        
        # 恢复成 2D 空间结构
        y_hat = out.transpose(1, 2).reshape(N_valid, C, H, W)
        return y_hat


# =====================================================================
# 模块 2: BYOL 风格 JEPA 辅助损失探针
# =====================================================================
class AuxMultimodalJEPABranch(nn.Module):
    def __init__(
        self,
        dim,
        reduction_ratio=2,
        ema_decay=0.996
    ):
        super().__init__()
        self.dim = dim
        self.ema_decay = ema_decay

        # 物理挖空锚点
        self.mask_token = nn.Parameter(torch.zeros(1, dim, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)

        # --------------------------------------------------
        # 投影头设计 (采用 GroupNorm 规避 Micro-batch 统计量崩塌)
        # --------------------------------------------------
        self.student_proj = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=False),
            nn.GroupNorm(32, dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GroupNorm(32, dim)
        )

        self.teacher_proj = copy.deepcopy(self.student_proj)
        self.teacher_proj.eval() # 老师永远不动
        for p in self.teacher_proj.parameters():
            p.requires_grad = False

        self.predictor = SRA_Predictor(
            dim, reduction_ratio=reduction_ratio
        )

    @torch.no_grad()
    def _update_teacher(self):
        """严格的 EMA 动量更新，包含权重与 GroupNorm Buffers"""
        for ps, pt in zip(self.student_proj.parameters(), self.teacher_proj.parameters()):
            pt.data.mul_(self.ema_decay)
            pt.data.add_((1 - self.ema_decay) * ps.data)

        for bs, bt in zip(self.student_proj.buffers(), self.teacher_proj.buffers()):
            bt.copy_(bs)

    def forward(self, feat_cv3, W, m_text, batch_dict):
        # AMP/BF16 兼容的安全返回，避免返回浮点数 0 导致的数据类型崩溃
        if not self.training:
            return feat_cv3.new_tensor(0.0)

        B, C, H, Wf = feat_cv3.shape
        device = feat_cv3.device
        nc = W.shape[0] // B

        # --------------------------------------------------
        # 1. Teacher 靶标生成与 Stop-Gradient
        # --------------------------------------------------
        with torch.no_grad():
            self._update_teacher()
            target_feat = self.teacher_proj(feat_cv3)
            # 靶标投影至单位超球面，为后续 Cosine Loss 奠定基石
            target_feat = F.normalize(target_feat, dim=1)

        # --------------------------------------------------
        # 2. 向量化解析 BBox 与有效性过滤
        # --------------------------------------------------
        bboxes = batch_dict["bboxes"]
        batch_idx = batch_dict["batch_idx"].flatten()
        cls_idx = batch_dict["cls"].flatten()

        cx, cy, bw, bh = bboxes.unbind(-1)

        # 换算至绝对坐标并截断越界
        x1 = ((cx - bw / 2) * Wf).clamp(0, Wf).long()
        y1 = ((cy - bh / 2) * H).clamp(0, H).long()
        x2 = ((cx + bw / 2) * Wf).clamp(0, Wf).long()
        y2 = ((cy + bh / 2) * H).clamp(0, H).long()

        # 严格屏蔽负样本 (cls=-1) 和异常尺寸框
        valid_mask = (x2 > x1) & (y2 > y1) & (cls_idx >= 0)

        if valid_mask.sum() == 0:
            return feat_cv3.new_tensor(0.0)

        # 提取 N_valid 级别张量
        b_idx = batch_idx[valid_mask]
        c_idx = cls_idx[valid_mask]
        x1, y1, x2, y2 = x1[valid_mask], y1[valid_mask], x2[valid_mask], y2[valid_mask]
        # N_valid = len(b_idx)

        # x_valid = feat_cv3[b_idx]
        # target_valid = target_feat[b_idx]

        # 文本精准提取
        text_idx = (b_idx * nc + c_idx).long()
        # ====================================================================
        # 语义级掩码合并 (Semantic Mask Merging)
        # 找出当前 Batch 中所有【不重复】的文本提示，并返回原始框的映射关系
        # unique_text_idx 形状: [N_merged] (合并后的真实实例数)
        # inverse_indices 形状: [N_all_boxes] (告诉我们每个原框属于哪个合并后的组)
        # ====================================================================
        unique_text_idx, inverse_indices = torch.unique(text_idx, return_inverse=True)
        N_merged = len(unique_text_idx)
        # 注意：这里我们需要根据 unique_text_idx 倒推属于哪张图
        # b_idx_merged 就是去重后对应的图像 batch 索引
        b_idx_merged = unique_text_idx // nc

        x_merged = feat_cv3[b_idx_merged]          # [N_merged, C, H, W]
        target_merged = target_feat[b_idx_merged]  # [N_merged, C, H, W]

        # 提取去重后的唯一文本特征
        w_seq_merged = W[unique_text_idx]          # [N_merged, L, C]
        m_seq_merged = m_text[unique_text_idx]     # [N_merged, L]
        # ====================================================================
        # 向量化构建多洞掩码 (Multi-Hole Spatial Mask)
        # ====================================================================
        # 初始化合并后的掩码
        spatial_mask = torch.zeros((N_merged, 1, H, Wf), device=device)
        
        # 将原来所有的框，按照 inverse_indices 映射，"画"进合并后的掩码里
        # 如果多个框的 inverse_indices 相同（即对应同一个文本），它们会被画在同一个矩阵里
        for i in range(len(b_idx)): # 这里的循环次数是原框数，极快
            group_id = inverse_indices[i]
            # 使用 = 1.0 (即使重叠了也依然是 1.0)
            spatial_mask[group_id, 0, y1[i]:y2[i], x1[i]:x2[i]] = 1.0

        # 对齐数据类型
        spatial_mask = spatial_mask.to(feat_cv3.dtype)

        # --------------------------------------------------
        # 3. 极速向量化空间掩码生成 (核心性能优化点)
        # --------------------------------------------------
        x_student = self.student_proj(x_merged)

        # yy = torch.arange(H, device=device)[None, :, None] # [1, H, 1]
        # xx = torch.arange(Wf, device=device)[None, None, :] # [1, 1, W]

        # # 利用张量广播瞬间生成所有实例的二维遮挡矩阵
        # spatial_mask = (
        #     (yy >= y1[:, None, None]) & 
        #     (yy < y2[:, None, None]) & 
        #     (xx >= x1[:, None, None]) & 
        #     (xx < x2[:, None, None])
        # )

        # # 对齐数据类型，兼容 AMP 混合精度训练
        # spatial_mask = spatial_mask.unsqueeze(1).to(x_student.dtype)

        # 物理挖空并注入噪声锚点，维持局部方差稳定
        noise = torch.randn_like(x_student) * 0.02
        x_m = x_student * (1 - spatial_mask) + (self.mask_token + noise) * spatial_mask

        # --------------------------------------------------
        # 4. 预测与混合目标对齐 Loss
        # --------------------------------------------------
        pred = self.predictor(x_m, w_seq_merged, m_seq_merged)
        # 将 [N_valid, 1, H, W] 转换为布尔掩码 [N_valid, H, W]
        mask_bool = spatial_mask.squeeze(1).bool()
        # ====================================================================
        # 只提取在 BBox 洞内的像素！
        # .permute 把通道 C 放到最后: [N_valid, H, W, C]
        # [mask_bool] 索引会瞬间摧毁空间维度，把所有 True 的像素展平为 2D 矩阵！
        # 此时 pred_sparse 的形状暴降为 [M, C] (M 是所有框内有效像素的总和)
        # ====================================================================
        pred_sparse = pred.permute(0, 2, 3, 1)[mask_bool]
        target_sparse = target_merged.permute(0, 2, 3, 1)[mask_bool]

        # 安全防线：如果框的面积太小，有效像素为 0
        if pred_sparse.shape[0] == 0:
            return pred.new_tensor(0.0)
        
        pred_sparse = F.normalize(pred_sparse, dim=-1)
        target_sparse = F.normalize(target_sparse, dim=-1)

        # (a) Cosine Loss
        # 因为我们已经提纯了有效像素，直接用 .mean() 即可，不需要再乘 spatial_mask 了！
        cosine_loss = 1 - F.cosine_similarity(pred_sparse, target_sparse, dim=-1)
        cosine_loss = cosine_loss.mean()

        # (b) Smooth L1 损失: 在超球面上提供平滑的梯度约束，惩罚局部尖锐震荡
        l1_loss = F.smooth_l1_loss(pred_sparse, target_sparse, reduction="mean")

        # 混合加权
        loss = cosine_loss + 0.5 * l1_loss
        return loss