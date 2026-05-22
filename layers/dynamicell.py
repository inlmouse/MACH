import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiDynamicCell(nn.Module):
    def __init__(self, in_dim, t_dim, k=3, stride=1, hidden_ratio=2):
        super().__init__()
        self.in_dim = in_dim
        self.k = k
        self.stride = stride

        hidden_dim = int(t_dim * hidden_ratio)

        # --- 动态部分：生成 Δkernel ---
        self.spat_gen = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim * k * k)
        )

        # --- 静态 Point-wise ---
        self.pw_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=True)

        # --- 初始化 ---
        self._init_identity_kernel()

    def _init_identity_kernel(self):
        """
        初始化为 identity kernel:
        Δw = 0
        实际 kernel = identity
        """
        nn.init.zeros_(self.spat_gen[-1].weight)

        bias = torch.zeros(self.in_dim * self.k * self.k)
        center = (self.k * self.k) // 2

        for c in range(self.in_dim):
            bias[c * self.k * self.k + center] = 1.0

        with torch.no_grad():
            self.spat_gen[-1].bias.copy_(bias)

    def forward(self, v_feat, t_query):
        """
        v_feat: [B, C, H, W]
        t_query: [B, 1, t_dim]
        """
        B, C, H, W = v_feat.shape

        # --- 1. 生成 kernel (identity + Δw) ---
        w = self.spat_gen(t_query)  # [B, C*k*k]
        w = w.view(B * C, 1, self.k, self.k)

        # --- 2. depth-wise conv ---
        v_in = v_feat.view(1, B * C, H, W)

        v_spat = F.conv2d(
            v_in,
            w,
            stride=self.stride,
            padding=self.k // 2,
            groups=B * C
        )

        H_out, W_out = v_spat.shape[-2:]
        v_spat = v_spat.view(B, C, H_out, W_out)

        # --- 5. 通道融合 ---
        v_out = self.pw_conv(v_spat)

        return v_out