import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class GaLN(nn.Module):
    """
    GaLN: Gaussian-modulated LayerNorm for Grounding Tasks
    
    核心设计：
    - gamma 直接来自 MRL 文本特征（不允许任何投影）
    - beta 是全局可学习参数（与标准 LayerNorm 的 bias 一致）
    - 局部路径仅做增量调制：γ ⊙ G ⊙ x_norm（去掉多余的 +1）
    - 同一 label 的 bbox 使用 softmax + tau 加权融合生成一个 mask
    - 向量化优化：减少 Python 循环层数，按 batch + unique label 处理
    - 支持 bbox dropout（训练时模拟缺失标注 / CFG）
    """
    def __init__(
        self,
        dim: int,                    # 当前特征图通道数 C
        dropout_p: float = 0.5,      # bbox dropout 概率
        alpha: float = 0.1,          # 局部残差权重
        k: float = 2.0,              # Gaussian Mask 内部平滑程度（越大越平坦）
        tau: float = 0.3,            # softmax temperature
        learnable_alpha: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.dropout_p = dropout_p
        self.k = k
        self.tau = tau
        self.eps = eps

        # beta 是全局可学习参数
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

        # 全局残差 alpha
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.alpha = alpha

        self.layer_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # 缓存 meshgrid：key = (feat_h, feat_w, stride)
        self._grid_cache = {}   # {(h, w, stride): (y_grid, x_grid)}

        nn.init.zeros_(self.beta)

    def _get_grids(self, feat_h: int, feat_w: int, device):
        cache_key = (feat_h, feat_w)
        if cache_key not in self._grid_cache:
            y_grid, x_grid = torch.meshgrid(
                torch.arange(feat_h, device=device, dtype=torch.float32),
                torch.arange(feat_w, device=device, dtype=torch.float32),
                indexing='ij'
            )
            self._grid_cache[cache_key] = (y_grid, x_grid)
        return self._grid_cache[cache_key]

    #@staticmethod
    def _generate_gaussian_mask(
        self,
        bboxes: torch.Tensor,      # [M, 4] 归一化坐标 (0~1)
        feat_h: int,               # 特征图高度
        feat_w: int,               # 特征图宽度
        # stride: int,             # ← 当 bbox 已归一化时，可以移除或忽略
    ) -> torch.Tensor:
        """生成单个类别的 Gaussian mask（bboxes 为归一化坐标）"""
        device = bboxes.device
        M = bboxes.shape[0]

        # 将归一化坐标映射到特征图坐标系
        # 假设 bboxes 是 [x1, y1, x2, y2] 格式（最常见）
        bboxes_feat = bboxes.clone()
        bboxes_feat[:, [0, 2]] *= feat_w      # x 坐标缩放到特征图宽度
        bboxes_feat[:, [1, 3]] *= feat_h      # y 坐标缩放到特征图高度

        cx = (bboxes_feat[:, 0] + bboxes_feat[:, 2]) / 2
        cy = (bboxes_feat[:, 1] + bboxes_feat[:, 3]) / 2
        bw = (bboxes_feat[:, 2] - bboxes_feat[:, 0]).clamp(min=1e-4)
        bh = (bboxes_feat[:, 3] - bboxes_feat[:, 1]).clamp(min=1e-4)

        sigma_x = bw / self.k
        sigma_y = bh / self.k

        # 使用缓存的 grid
        y_grid, x_grid = self._get_grids(feat_h, feat_w, device)
        y_grid = y_grid.unsqueeze(0).expand(M, -1, -1)
        x_grid = x_grid.unsqueeze(0).expand(M, -1, -1)

        # 高斯分布
        g = torch.exp(
            -((x_grid - cx.unsqueeze(1).unsqueeze(2))**2) / (2 * sigma_x.unsqueeze(1).unsqueeze(2)**2) -
            ((y_grid - cy.unsqueeze(1).unsqueeze(2))**2) / (2 * sigma_y.unsqueeze(1).unsqueeze(2)**2)
        )

        # softmax 加权融合
        logits = g / self.tau
        weights = torch.softmax(logits.flatten(1), dim=0).view_as(g)
        fused_mask = (weights * g).sum(dim=0)

        fused_mask = fused_mask / (fused_mask.max() + 1e-8)
        
        return fused_mask.unsqueeze(0).unsqueeze(0).contiguous()   # [1, 1, H, W]

    def _generate_all_gaussians(self, bboxes, H, W):
        """
        向量化 Gaussian 生成
        bboxes: [N, 4] (normalized)
        return: [N, H, W]
        """
        device = bboxes.device
        N = bboxes.shape[0]

        if N == 0:
            return torch.zeros(0, H, W, device=device)

        b = bboxes.clone()
        b[:, [0, 2]] *= W
        b[:, [1, 3]] *= H

        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        bw = (b[:, 2] - b[:, 0]).clamp(min=1e-4)
        bh = (b[:, 3] - b[:, 1]).clamp(min=1e-4)

        sigma_x = bw / self.k
        sigma_y = bh / self.k

        y, x = self._get_grids(H, W, device)

        x = x.unsqueeze(0)  # [1,H,W]
        y = y.unsqueeze(0)

        cx = cx.view(N, 1, 1)
        cy = cy.view(N, 1, 1)
        sx = sigma_x.view(N, 1, 1)
        sy = sigma_y.view(N, 1, 1)

        g = torch.exp(
            -((x - cx) ** 2) / (2 * sx ** 2)
            -((y - cy) ** 2) / (2 * sy ** 2)
        )

        return g  # [N,H,W]
    
    def forward(
        self,
        x: torch.Tensor,                    # [B, C, H, W]
        text_feats: torch.Tensor,           # [B, num_classes, D]  D >= C 的原始 MRL 特征
        batch: dict,                        # 包含 'bboxes', 'cls', 'batch_idx' 等键
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # LayerNorm + global beta
        x_norm = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        global_out = x_norm + self.beta#.view(1, C, 1, 1)

        # 空输入 或 训练时 dropout（模拟缺失标注）
        if text_feats is None:
            return global_out
        if self.training and torch.rand((), device=device) < self.dropout_p:
            return global_out

        bboxes=batch['bboxes']                 # [N, 4] normlized bbox coordinates [x1, y1, x2, y2]
        labels=batch['cls'].view(-1).long()    # [N] 类别索引
        batch_idx=batch['batch_idx'].long()    # [N] 每个 bbox 所属的 batch index

        N = bboxes.shape[0]

        # =============================
        # 一次性生成所有 Gaussian
        # =============================
        g = self._generate_all_gaussians(bboxes, H, W)  # [N,H,W]

        # ===== Step 2: group id =====
        num_classes = text_feats.shape[1]
        #group_id = batch_idx * num_classes + labels  # [N]
        group_key = torch.stack([batch_idx, labels], dim=1)
        unique_gid, group_id = torch.unique(
            group_key,
            dim=0,
            return_inverse=True
        )

        # ===== Step 3: density-invariant aggregation=====
        # DeepSets / PointNet style aggregation
        fused = g  # [N, H, W]  每个 bbox 等权

        # sum aggregation
        group_mask = torch.zeros(
            int(group_id.max()) + 1,
            H, W,
            device=device,
            dtype=x.dtype
        )

        group_mask = group_mask.index_add(0, group_id, fused)

        # count aggregation（关键：防止 density bias）
        count = torch.zeros(
            group_mask.shape[0],
            device=device,
            dtype=x.dtype
        )

        count = count.index_add(0, group_id, torch.ones(N, device=device))

        # mean pooling（核心替换点）
        group_mask = group_mask / (count[:, None, None] + 1e-6)

        # =============================
        # 还原 (batch, label)
        # =============================
        group_batch = unique_gid[:, 0]
        group_label = unique_gid[:, 1]

        out_mask = torch.zeros(
            B, num_classes, H, W,
            device=device,
            dtype=x.dtype
        )

        out_mask[group_batch, group_label] = group_mask

        # =============================
        # normalize（每类独立）
        # =============================
        #out_mask = out_mask / (
        #    out_mask.amax(dim=(2,3), keepdim=True) + 1e-8
        #)
        # text_feats: [B,K,D] → 截断到 C
        text_feats = text_feats[:, :, :C]

        # gamma: [B,C,K]
        gamma = text_feats.permute(0, 2, 1)

        # [B,C,H,W]
        gamma_map = torch.einsum('bck,bkhw->bchw', gamma, out_mask)

        # final
        out = global_out + self.alpha * (gamma_map * x_norm)

        return out

        # 直接截取前 C 维作为 gamma
        text_feats = text_feats[:, :, :C]   # [B, num_classes, C]

        # 输出初始化为全局结果
        out = global_out.clone()

        # ================= 向量化优化（按 batch + unique label）=================
        for b in range(B):
            mask_b = (batch_idx == b)
            if not mask_b.any():
                continue

            bboxes_b = bboxes[mask_b]      # [Mb, 4]
            labels_b = labels[mask_b].squeeze(-1).long()      # [Mb]  squeeze to 1D and ensure long dtype

            # 获取当前 batch 的 unique labels 及其 inverse（用于快速分组）
            unique_labels, inverse = torch.unique(labels_b, return_inverse=True)
            num_unique = unique_labels.shape[0]

            if num_unique == 0:
                continue

            local_mod = torch.zeros((1, C, H, W), device=device, dtype=x.dtype)

            # 对每个 unique label 生成 mask 并累加
            for i in range(num_unique):
                lab = unique_labels[i].item()  # Convert to Python int for indexing
                lab_mask = (inverse == i)
                lab_boxes = bboxes_b[lab_mask]                     # [M_lab, 4]

                # 生成 Gaussian mask
                G = self._generate_gaussian_mask(lab_boxes, H, W)  # [1, 1, H, W]

                # gamma 来自文本特征（直接索引）
                gamma = text_feats[b, lab]                         # [C]
                gamma_map = gamma.view(1, C, 1, 1) * G             # [1, C, H, W]

                delta = gamma_map * x_norm[b:b+1]                  # [1, C, H, W]
                local_mod += delta

            # 应用局部残差调制
            out[b:b+1] = global_out[b:b+1] + self.alpha * local_mod

        return out




class SeMoLN(nn.Module):
    """
    SeMoLN: Semantic Modulated LayerNorm for Grounding

    Design goals:
    1. Keep LayerNorm-style normalization for stable feature dynamics.
    2. Use raw text features directly as channel-wise modulation signals
       (no projection layer, only channel truncation).
    3. Only valid categories in the current batch participate in modulation,
       avoiding noisy semantic interference from absent labels.
    4. Replace class-wise softmax with independent sigmoid activation,
       which is more suitable for multi-label grounding tasks.
    5. Use residual modulation:
           output = normalized_feature + beta + alpha * (gamma * normalized_feature)

    Input:
        x:          [B, C, H, W] visual feature map
        text_feats: [B, K, D] text features from MRL encoder
                    K = number of classes
                    D >= C
        batch:
            {
                'batch_idx': [N] bbox-to-image index
                'cls':       [N] bbox class labels
            }

    Output:
        out: [B, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        smstrength: float = 0.1,
        tau: float = 0.3,
        semantic_modulate: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.semantic_modulate = semantic_modulate

        self.spa_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        # Temperature for similarity activation
        self.tau = nn.Parameter(torch.tensor(float(tau)))

        # LayerNorm-style learnable weights and bias
        self.beta = nn.Parameter(torch.zeros(dim))
        self.alpha = nn.Parameter(torch.tensor(float(1.0)))

        # Residual modulation strength
        self.smstrength = nn.Parameter(torch.tensor(float(smstrength)))

    def _build_valid_mask(self, batch, B, K, device):
        """
        Build semantic validity mask.

        Only categories that actually appear in the current batch
        are allowed to participate in feature modulation.

        Returns:
            valid_mask: [B, K] bool
        """
        valid_mask = torch.zeros(
            (B, K),
            device=device,
            dtype=torch.bool,
        )

        if batch is None:
            return valid_mask

        if "batch_idx" not in batch or "cls" not in batch:
            return valid_mask

        b_idx = batch["batch_idx"].long()
        l_idx = batch["cls"].view(-1).long()

        if len(b_idx) > 0:
            valid_mask[b_idx, l_idx] = True

        return valid_mask

    def forward(
        self,
        x: torch.Tensor,
        text_feats: torch.Tensor = None,
        batch: dict = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        B, C, H, W = x.shape
        device = x.device

        # --------------------------------------------------
        # Step 1: LayerNorm-style normalization
        # --------------------------------------------------
        x = self.spa_proj(x)  # 可选的空间投影层
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Base output
        out = self.alpha * x_norm + self.beta.view(1, C, 1, 1)

        # No text condition -> pure normalized output
        if text_feats is None:
            return out, None

        # --------------------------------------------------
        # Step 2: Align text feature dimension with stage channel
        # --------------------------------------------------
        if text_feats.ndim == 2:
            text_feats = text_feats.unsqueeze(0)  # [1, K, D]
        K = text_feats.shape[1]
        text_feats = text_feats[:, :, :C]  # [B, K, C]

        # --------------------------------------------------
        # Step 3: Semantic validity mask
        # --------------------------------------------------
        if self.training:
            valid_mask = self._build_valid_mask(batch, B, K, device)
        else:
            valid_mask = torch.ones(
            (B, K),
            device=device,
            dtype=torch.bool,
        )

        # If no valid label exists, skip modulation
        if not valid_mask.any():
            return out, None

        # --------------------------------------------------
        # Step 4: Visual-text similarity map
        # --------------------------------------------------
        # Normalize for cosine similarity
        visual_feat = F.normalize(x_norm, dim=1)      # [B, C, H, W]
        text_feat = F.normalize(text_feats, dim=2)    # [B, K, C]

        # Similarity map: [B, K, H, W]
        sim_map = torch.einsum(
            "bchw,bkc->bkhw",
            visual_feat,
            text_feat,
        )

        if not self.semantic_modulate:
            return out, sim_map

        # Mask invalid categories
        valid_mask = valid_mask.view(B, K, 1, 1)
        sim_map_masked = sim_map.masked_fill(~valid_mask, -20.0)

        # --------------------------------------------------
        # Step 5: Independent sigmoid activation
        # --------------------------------------------------
        # Important:
        # Use sigmoid instead of softmax.
        # Grounding is multi-label rather than mutually exclusive.
        active_sim = torch.sigmoid(sim_map_masked / self.tau)

        # --------------------------------------------------
        # Step 6: Channel-wise semantic modulation
        # --------------------------------------------------
        # gamma: [B, C, K]
        gamma = text_feats.permute(0, 2, 1)

        # gamma_map: [B, C, H, W]
        gamma_map = torch.einsum(
            "bck,bkhw->bchw",
            gamma,
            active_sim,
        )

        # --------------------------------------------------
        # Step 7: Residual modulation output
        # --------------------------------------------------
        out = out + self.smstrength * (gamma_map * x_norm)
        
        return out, sim_map

    

class SelfAttn(nn.Module):
    """
    SelfAttn: 
    """
    def __init__(
        self,
        dim: int,                    # 当前特征图通道数 C
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        #self.tau = nn.Parameter(torch.tensor(tau))
        self.eps = eps
        self.ffn = nn.Sequential(nn.Conv2d(dim, 2 * dim, kernel_size=1), nn.SiLU(), nn.Conv2d(2 * dim, dim, kernel_size=1))
        self.dropout_p = 0.5
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.ffn[-1].weight)
        #nn.init.constant_(self.ffn[-1].weight, 1e-6)
        nn.init.zeros_(self.ffn[-1].bias)

        self.projector = nn.Linear(dim, dim)
        nn.init.zeros_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def forward(
        self,
        x: torch.Tensor,                # [B, C, H, W] 视觉特征
        text_feats: torch.Tensor,       # [B, K, D] 原始 MRL 特征 (K为类数, D>=C)
        batch: dict = None,             # 此时主要用于获取类别信息或推理参考
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # 2. 条件跳过 (Training Robustness)
        if text_feats is None:
            return x
        
        
        if text_feats.ndim == 2:
            text_feats = text_feats.unsqueeze(0)  # [1, K, D]
        K = text_feats.shape[1]  # 类别数==num_classes
        

        # if self.training:
        #     with torch.no_grad():
        #         # valid_semantic_mask: [B, K]
        #         valid_semantic_mask = torch.zeros((B, K), device=device, dtype=torch.bool)
        #         # 获取当前 batch 中所有正样本的 batch_id 和 label_id
        #         b_idx = batch['batch_idx'].long()
        #         l_idx = batch['cls'].view(-1).long()
        #         valid_semantic_mask[b_idx, l_idx] = 1.0
        #     valid_K = max(l_idx).item() + 1
        #     valid_semantic_mask = valid_semantic_mask[:, :valid_K]  # [B, valid_K]
        # else:
        #     valid_K = K
        #     valid_semantic_mask = torch.ones((B, K), device=device, dtype=torch.bool)
        valid_K = K

        text_feats_sub = text_feats[:, :valid_K, :C]
        text_feats_sub = text_feats_sub + self.projector(text_feats_sub)  # [B, valid_K, C]
        x_feat = F.normalize(x, dim=1)  # [B, C, H, W]
        t_feat = F.normalize(text_feats_sub, dim=2)  # [B, K, C]

        # sim_map: [B, K, H, W] - 衡量每个空间位置与 K 个类别的匹配度
        sim_map = torch.einsum('bchw,bkc->bkhw', x_feat, t_feat)
        #mask = valid_semantic_mask.view(B, valid_K, 1, 1)  # [B,K,1,1]

        #sim_map_masked = sim_map.masked_fill(~mask.bool(), 0.0)
        active_sim = sim_map / (sim_map.abs().max() + self.eps)

        attnout = torch.einsum('bkhw,bck->bchw', active_sim, text_feats_sub.permute(0, 2, 1))
        if self.training and torch.rand((), device=device) < self.dropout_p:
            return x + self.ffn(x)
        x = x + self.alpha * attnout
        x = x + self.ffn(x)

        return x



class FiLM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # 参数生成器：从聚合后的文本特征生成 gamma 和 beta
        self.generator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 2 * dim)
        )
        
        # 初始化为恒等变换
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(
        self,
        x: torch.Tensor,               # [B, C, H, W] 视觉特征
        text_feats: torch.Tensor,      # [B, K, D] 原始 MRL 特征
        batch: dict = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        K = text_feats.shape[1]
        device = x.device

        # 1. 维度对齐
        # 取 MRL 特征的前 C 维以匹配视觉通道数
        if text_feats.ndim == 2:
            text_feats = text_feats.unsqueeze(0)  # [1, K, D]
        t_feats = text_feats[:, :, :self.dim] # [B, K, C]

        # 2. 聚合文本特征 (t_avg: [B, C])
        if self.training and batch is not None:
            # --- 训练模式：只聚合当前 batch 中正样本的特征 ---
            # 创建 mask: [B, K]
            mask = torch.zeros((B, K), device=device)
            b_idx = batch['batch_idx'].long()
            l_idx = batch['cls'].view(-1).long()
            mask[b_idx, l_idx] = 1.0
            
            # 计算每个 batch 的正样本均值
            # 增加 eps 防止某个 batch 没有任何正样本导致除零
            denom = mask.sum(dim=1, keepdim=True) + 1e-8
            t_avg = (t_feats * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            # --- 推理模式：通常聚合所有查询类别的特征 ---
            # 也可以根据需求修改为聚合前 K 个有效类
            t_avg = t_feats.mean(dim=1)

        # 3. 生成 FiLM 参数 [B, 2*C]
        params = self.generator(t_avg)
        gamma, beta = params.chunk(2, dim=1) # 各自为 [B, C]

        # 4. 维度对齐并应用变换
        # x_mod = (1 + gamma) * x + beta
        gamma = gamma.view(-1, C, 1, 1)
        #gamma = torch.tanh(gamma) * 1.0
        beta = beta.view(-1, C, 1, 1)

        x = (1.0 + gamma) * x.clone() + beta

        #LayerNorm
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + 1e-6)

        return x
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Tuple

class LiteCrossMLA(nn.Module):
    r"""Lightweight multi-scale linear cross attention
    Q: from image feature x [B, C, H, W]
    K, V: from text feature w [B, L, C]
    """

    def __init__(
        self,
        in_channels: int,      # 图像输入通道
        context_dim: int,      # 文本特征通道 (C)
        out_channels: int,     # 输出通道
        heads: int = 1,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: Tuple[int, ...] = (5,),
        eps=1.0e-6,
    ):
        super(LiteCrossMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        self.dim = dim
        self.heads = heads

        # Q 投影 (来自图像 x)
        self.q_proj = nn.Conv2d(in_channels, total_dim, 1, bias=use_bias)
        
        # KV 投影 (来自文本 w) - 使用 Linear 层处理 [B, L, C]
        self.kv_proj = nn.Linear(context_dim, 2 * total_dim, bias=use_bias)

        # 多尺度聚合模块 (作用于图像 Q，捕捉不同感受野的查询)
        self.q_aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(total_dim, total_dim, scale, padding=scale//2, groups=total_dim, bias=use_bias),
                nn.Conv2d(total_dim, total_dim, 1, groups=heads, bias=use_bias),
            ) for scale in scales
        ])

        self.kernel_func = nn.ReLU(inplace=False) if kernel_func == "relu" else nn.Identity()

        # 最终投影层：输入包含 (1 + len(scales)) 个尺度的拼接结果
        self.proj = nn.Sequential(
            nn.Conv2d(total_dim * (1 + len(scales)), out_channels, 1, bias=use_bias),
            nn.BatchNorm2d(out_channels) if norm[1] == "bn2d" else nn.Identity()
        )
        nn.init.constant_(self.proj[-1].weight, 1e-6)
        nn.init.constant_(self.proj[-1].bias, 0.0)

    @torch.amp.autocast('cuda', enabled=False)
    def relu_linear_cross_att(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: [B, heads * (1+scales), dim, H*W]
        k: [B, heads, dim, L]
        v: [B, heads, dim, L]
        """
        q, k, v = q.float(), k.float(), v.float()
        B, C_q, H, W = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
        L = k.shape[1] # 文本长度

        # 重塑 Q (包含多尺度拼接后的维度)
        q = q.reshape(B, -1, self.dim, H * W).transpose(-1, -2) # [B, heads_multi, HW, dim]
        
        # 重塑 K, V (文本)
        k = k.reshape(-1, L, self.heads, self.dim).permute(0, 2, 3, 1) # [B, heads, dim, L]
        v = v.reshape(-1, L, self.heads, self.dim).transpose(1, 2)     # [B, heads, L, dim]

        # 线性注意力核函数
        q = self.kernel_func(q) + 1e-6
        k = self.kernel_func(k) + 1e-6 # k 现在是 [B, heads, dim, L]

        # 计算线性注意力: (Q @ (K.T @ V))
        # 注意：这里 Q 可能是多尺度的，需要对齐 heads 维度
        # 为了简化，我们假设多尺度是在 head 维度叠加的
        # K, V 需要扩展以匹配 Q 的多尺度数量
        num_scales = q.shape[1] // self.heads
        if num_scales > 1:
            k = k.repeat(1, num_scales, 1, 1) # [B, heads_multi, dim, L]
            v = v.repeat(1, num_scales, 1, 1) # [B, heads_multi, L, dim]

        # v 加上一行全 1 用于归一化 (linear attention trick)
        v_padded = F.pad(v, (0, 1), mode="constant", value=1) # [B, heads_multi, L, dim+1]
        
        # kv: [B, heads_multi, dim, dim+1]
        kv = torch.matmul(k, v_padded)
        
        # out: [B, heads_multi, HW, dim+1]
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = out.transpose(-1, -2).reshape(B, -1, H, W)
        return out

    def forward(self, x: torch.Tensor, w: torch.Tensor, batch: dict = None) -> torch.Tensor:
        """
        x: image feature [B, in_channels, H, W]
        w: text feature [B, K, context_dim]
        """

        if w.ndim == 2:
            w = w.unsqueeze(0)  # [1, K, D]
        

        # 1. 生成图像 Q 及其多尺度版本
        q_base = self.q_proj(x)
        q_list = [q_base]
        for op in self.q_aggreg:
            q_list.append(op(q_base))
        q_multi = torch.cat(q_list, dim=1) # [B, total_dim * (1 + len(scales)), H, W]
        q_multi = F.normalize(q_multi, p=2, dim=1, eps=1e-6)

        # 2. 生成文本 K, V
        kv_text = self.kv_proj(w) # [B, L, 2 * total_dim]
        k_text, v_text = torch.split(kv_text, kv_text.size(-1) // 2, dim=-1)
        k_text = F.normalize(k_text, p=2, dim=-1, eps=1e-6)
        # 为了极致的数值稳定，归一化 V 也是可以的, but可能会导致输出特征的表达力受限
        # v_text = F.normalize(v_text, p=2, dim=-1, eps=1e-6)

        # 3. 计算线性交叉注意力
        out = self.relu_linear_cross_att(q_multi, k_text, v_text)
        
        # 4. 最终投影
        out = self.proj(out)
        return x + out