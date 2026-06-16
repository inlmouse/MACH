import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TCEM(nn.Module):
    """
    Text-Conditioned Energy Minimization (TCEM) Module
    
    核心贡献:
    1. Hard Active Graph Pruning: 提取高响应视觉/文本节点，极大降低图推断复杂度。
    2. TCPF (Text-Conditioned Potential Function): 利用 PSD 兼容矩阵稳定类间语义博弈。
    3. Learnable Mean-field Unfolding: 引入非负约束的动态温度与平滑权重，实现自适应展开。
    """
    def __init__(self, dim, num_iterations=3, num_active_tokens=-1, num_active_texts=-1):
        super().__init__()
        self.num_active_tokens = num_active_tokens
        self.num_active_texts = num_active_texts
        self.num_iterations = num_iterations
        if num_active_tokens >=0 and num_active_texts >= 0:
            self.use_pruning = True
        else:
            self.use_pruning = False
        # 为了保证 Softplus 后的初始值与物理直觉一致，使用逆 Softplus 初始化
        # inverse_softplus(y) = math.log(math.exp(y) - 1)
        init_sigma = math.log(math.exp(1.0) - 1.0)
        init_lambda = math.log(math.exp(0.1) - 1.0)
        init_tau = math.log(math.exp(1.0) - 1.0)

        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        # ==========================================
        # 1. 视觉图构建参数 (带非负保护)
        # ==========================================
        self.feat_proj = nn.Linear(dim, dim)
        self.raw_sigma = nn.Parameter(torch.tensor([init_sigma]))
        
        # ==========================================
        # 2. 文本感知背景场 (Contextual Background Field)
        # ==========================================
        self.bg_head = nn.Sequential(
            nn.Linear(dim + 1, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

        # ==========================================
        # 3. TCPF: 文本条件势函数
        # ==========================================
        self.W_comp = nn.Linear(dim, dim, bias=False)
        self.bg_self_comp = nn.Parameter(torch.ones(1)) 
        
        # ==========================================
        # 4. Amortized Mean-field 参数 (严格非负约束)
        # ==========================================
        # self.raw_lambda_list = nn.ParameterList([
        #     nn.Parameter(torch.tensor([init_lambda])) for _ in range(num_iterations)
        # ])
        self.raw_lambda_list = nn.ParameterList([
            nn.Parameter(torch.tensor(-3.0)) for _ in range(num_iterations)
        ])
        # self.raw_tau_list = nn.ParameterList([
        #     nn.Parameter(torch.tensor([init_tau])) for _ in range(num_iterations)
        # ])
        self.raw_tau_list = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)),  # t=0, softplus(0.5) ≈ 0.97
            nn.Parameter(torch.tensor(-0.2)), # t=1, softplus(-0.2) ≈ 0.59
            nn.Parameter(torch.tensor(-1.0))  # t=2, softplus(-1.0) ≈ 0.31
        ])
        with torch.no_grad():
            # 假设 self.W_comp 是 nn.Linear(d, d, bias=False)
            nn.init.eye_(self.W_comp.weight)  # 初始化为单位矩阵
            self.W_comp.weight.data.mul_(0.1) # 缩小初始缩放，防止排斥力过猛
            
            # 同样处理特征投影层 self.feat_proj
            nn.init.eye_(self.feat_proj.weight)
            self.feat_proj.weight.data.mul_(0.1)
        self.raw_sigma = nn.Parameter(torch.tensor(0.2))
        self.bg_self_comp = nn.Parameter(torch.tensor(0.5)) # 或者一个适中的正数

    def forward(self, visual_feats, w, m):
        B, d, H, Ws = visual_feats.shape
        visual_feats = visual_feats.view(B, d, H * Ws)
        visual_feats = visual_feats.permute(0, 2, 1)
        N = H*Ws
        if m is not None:
            _, L, _ = w.shape
            w = w.reshape(B, -1, L, d)
            m = m.reshape(B, -1, L)
            text_feats = self._get_last_token_pooling(w, m)
        else:
            text_feats = w
        
        M = text_feats.shape[1]
        device = visual_feats.device

        # 1. 特征归一化
        visual_feats = F.normalize(visual_feats, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        A_unary = torch.bmm(visual_feats, text_feats.transpose(1, 2))

        # A_unary = A_unary.transpose(1, 2).view(B, M, H, Ws).contiguous()
        # return A_unary * self.logit_scale.exp() + self.bias

        # 2. 逻辑分支：硬剪枝 vs 全图推断
        if self.use_pruning:
            k_tokens = min(self.num_active_tokens, N)
            k_texts = min(self.num_active_texts, M)
            
            # 执行 Top-K 采样
            max_spatial_scores, _ = A_unary.max(dim=-1)
            _, active_token_idx = torch.topk(max_spatial_scores, k_tokens, dim=-1)
            
            max_text_scores, _ = A_unary.max(dim=1)
            _, active_text_idx = torch.topk(max_text_scores, k_texts, dim=-1)
            
            # 提取活跃特征
            E_active = torch.gather(visual_feats, 1, active_token_idx.unsqueeze(-1).expand(-1, -1, d))
            T_active = torch.gather(text_feats, 1, active_text_idx.unsqueeze(-1).expand(-1, -1, d))
        else:
            k_tokens, k_texts = N, M
            active_token_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
            active_text_idx = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
            E_active, T_active = visual_feats, text_feats

        # 3. 构建能量组件 (U, mu, W) - 逻辑对于两分支完全通用
        A_active = torch.bmm(E_active, T_active.transpose(1, 2))
        max_active_resp, _ = A_active.max(dim=-1, keepdim=True)
        bg_energy = self.bg_head(torch.cat([E_active, max_active_resp], dim=-1))
        U = torch.cat([bg_energy, A_active], dim=-1)

        T_proj = self.W_comp(T_active)
        T_proj = T_proj - T_proj.mean(dim=1, keepdim=True)
        T_norm = F.normalize(T_proj, p=2, dim=-1)
        mu_text = torch.bmm(T_norm, T_norm.transpose(1, 2)) 

        mu = torch.zeros(B, k_texts + 1, k_texts + 1, device=device, dtype=E_active.dtype)
        mu[:, 1:, 1:] = mu_text
        mu[:, 0, 0] = self.bg_self_comp 

        sigma = F.softplus(self.raw_sigma) + 1e-4 
        E_proj = F.normalize(self.feat_proj(E_active), p=2, dim=-1)
        S = torch.bmm(E_proj, E_proj.transpose(1, 2))
        W_raw = torch.exp(0.5 * (S + S.transpose(1, 2)) / sigma)
        D_inv_sqrt = torch.rsqrt(W_raw.sum(dim=-1, keepdim=True) + 1e-6)
        W = W_raw * D_inv_sqrt * D_inv_sqrt.transpose(1, 2)

        # 4. 迭代展开
        Q_t = F.softmax(U, dim=-1)
        for t in range(self.num_iterations):
            lambda_t = F.softplus(self.raw_lambda_list[t])
            tau_t = F.softplus(self.raw_tau_list[t]) + 1e-4
            Q_t = F.softmax((U + lambda_t * torch.bmm(torch.bmm(W, Q_t), mu)) / tau_t, dim=-1)

        # 5. 还原与归一化
        # 使用 scatter 构建 Q_final，这种方式比索引赋值更具鲁棒性
        Q_final = torch.full((B, N, M + 1), 1e-4, device=device, dtype=E_active.dtype)
        
        # 统一处理背景项
        max_spatial_scores, _ = A_unary.max(dim=-1)
        Q_final[:, :, 0] = torch.sigmoid(self.bg_head(torch.cat([visual_feats, max_spatial_scores.unsqueeze(-1)], dim=-1))).squeeze(-1)

        # 将迭代结果 scatter 回去
        # 这里的 text_idx_with_bg 逻辑必须兼容 k_tokens == N 的情况
        idx_map = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.long), active_text_idx + 1], dim=1)
        # 使用 scatter_ 填充文本维
        Q_final.scatter_(2, idx_map.unsqueeze(1).expand(B, N, -1).gather(1, active_token_idx.unsqueeze(-1).expand(-1, -1, k_texts+1)), Q_t)

        # 归一化以严格满足单纯形约束
        Q_final = Q_final / (Q_final.sum(dim=-1, keepdim=True) + 1e-6)
        # return Q_final
        # 1. 剥离第 0 维的背景，保留纯文本类别概率空间 [B, N, M]
        Q_text = Q_final[:, :, 1:]

        # 2. 变换维度至标准的深度学习视觉格式 [B, M, H, W]
        # 先转置为 [B, M, N]，再完美还原为二维图像空间空间 [B, M, H, W]
        Q_out = Q_text.transpose(1, 2).view(B, M, H, Ws).contiguous()
        return Q_out * self.logit_scale.exp() + self.bias
    
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