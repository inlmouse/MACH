import torch
import torch.nn as nn
from .galn import GaLN, SeMoLN, SelfAttn, FiLM, LiteCrossMLA

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c = int(c2 * e)
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c*2, 1, bias=False), nn.BatchNorm2d(c*2), nn.SiLU())
        self.cv2 = nn.Sequential(nn.Conv2d((2+n)*c, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())
        self.bottleneck = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU(),
                nn.Conv2d(c, c, 3, 1, 1, groups=g, bias=False), nn.BatchNorm2d(c), nn.SiLU()
            ) for _ in range(n)
        ])
        self.shortcut = shortcut

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for b in self.bottleneck:
            r = y[-1]
            o = b(y[-1])
            y.append(o + r if self.shortcut and r.shape[1]==o.shape[1] else o)
        return self.cv2(torch.cat(y, 1))
    

class FPANeck(nn.Module):
    """
    Neck (FPN + PAN)
    - 依赖独立的 C2f 模块
    """
    def __init__(self, backbone_channels):
        super().__init__()
        c3, c4, c5 = backbone_channels  
        self.c3, self.c4, self.c5 = c3, c4, c5
        # self.c3norm = FiLM(c3)
        # self.c4norm = FiLM(c4)
        # self.c5norm = FiLM(c5)
        # self.c3mla = LiteCrossMLA(in_channels=c3, context_dim=768, out_channels=c3, heads=8, dim=8, scales=(3, 5))
        # self.c4mla = LiteCrossMLA(in_channels=c4, context_dim=768, out_channels=c4, heads=8, dim=8, scales=(5,))
        # self.c5mla = LiteCrossMLA(in_channels=c5, context_dim=768, out_channels=c5, heads=8, dim=8, scales=(5, 7))
        # ─────────────────────────────── Top-down ───────────────────────────────

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        # 6: C2f after up + concat P4
        self.c2f6 = C2f(c5 + c4, c4, n=3, shortcut=True, e=0.5)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # 9: C2f after up + concat P3
        self.c2f9 = C2f(c4 + c3, c3, n=3, shortcut=True, e=0.5)

        # ─────────────────────────────── Bottom-up ───────────────────────────────

        # 10: downsample from P3/8 (256 → 256, 3x3 stride=2)
        self.down1 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU()
        )

        # 12: C2f after down + concat previous fused P4
        self.c2f12 = C2f(c3 + c4, c4, n=3, shortcut=True, e=0.5)

        # 13: downsample from P4/16 (512 → 512, 3x3 stride=2)
        self.down2 = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU()
        )

        # 15: C2f after down + concat original P5
        self.c2f15 = C2f(c4 + c5, c5, n=3, shortcut=True, e=0.5)

    def get_output_channels(self):
        return [self.c3, self.c4, self.c5]  # P3/8 输出 c3，P4/16 输出 c4，P5/32 输出 c5

    def forward(self, features, w=None, batch=None):
        """
        Args:
            features: List[Tensor] = [P3/8 (256), P4/16 (512), P5/32 (1024)]
        Returns:
            List[Tensor]: [P3/8 (256), P4/16 (512), P5/32 (1024)]
        """
        p3, p4, p5 = features
        simp3 = simp4 = simp5 = None
        # p3 = self.c3norm(x=p3, text_feats=w, batch=batch)
        # p4 = self.c4norm(x=p4, text_feats=w, batch=batch)
        # p5 = self.c5norm(x=p5, text_feats=w, batch=batch)
        # Top-down path
        x = self.upsample1(p5)              # up P5
        x = torch.cat([x, p4], dim=1)       # concat backbone P4
        p4_fused = self.c2f6(x)

        x = self.upsample2(p4_fused)        # up fused P4
        x = torch.cat([x, p3], dim=1)       # concat backbone P3
        p3_out = self.c2f9(x)

        # Bottom-up path (PAN)
        x = self.down1(p3_out)              # down from P3
        x = torch.cat([x, p4_fused], dim=1) # concat fused P4
        p4_out = self.c2f12(x)

        x = self.down2(p4_out)              # down from fused P4
        x = torch.cat([x, p5], dim=1)       # concat original P5
        p5_out = self.c2f15(x)

        # p3_out = self.c3mla(p3_out, w, batch)
        # p4_out = self.c4mla(p4_out, w, batch)
        # p5_out = self.c5mla(p5_out, w, batch)

        if simp3 is not None and simp4 is not None and simp5 is not None:
            simp5 = self.upsample1(self.upsample2(simp5))  # up to P3 size
            simp4 = self.upsample1(simp4)  # up to P4 size
            # simp5 keeps original P5 size
            fused_simp = simp3 + simp4 + simp5  # 融合三个尺度的 simmap
            return [p3_out, p4_out, p5_out, fused_simp]


        return [p3_out, p4_out, p5_out, None]