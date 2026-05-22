import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.fpn import FPANeck
from layers.vitneck import MultiScaleViTFusion, ViTNeck
from layers.head import Detect
from loss.detectloss import DetectionLoss
from models.convnext import ConvNeXt

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        gc,
        ec,
        e=4
    ) -> None:
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class Residual(nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with large scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)
        
    def forward(self, x):
        w = x + self.m(x)
        return F.normalize(w, dim=-1, p=2)

class AdaptiveMemoryRectifier(nn.Module):
    """
    针对已对齐特征的语义微调模块
    原则：保真第一，增强第二
    """
    def __init__(self, dim, memory_blocks=128, r=4):
        super().__init__()
        # 使用低秩 (Low-rank) 投影，r 越小对原始特征的扰动越小
        self.latent_dim = dim // r
        
        # 仅使用线性投影，不加激活函数，保持特征空间的拓扑一致性
        self.down = nn.Linear(dim, self.latent_dim, bias=False)
        self.up = nn.Linear(self.latent_dim, dim, bias=False)
        
        # 记忆库：存储微小的语义偏差补丁
        self.mb = nn.Parameter(torch.randn(memory_blocks, self.latent_dim))
        nn.init.orthogonal_(self.mb) # 保持基向量的正交性
        
        # 零初始化门控：确保训练开始时 w_enhanced == w
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, w):
        """
        w: [B, K, C] -> 已经与图像对齐的 Qwen3 特征
        """
        # 1. 投影到低秩空间
        z = self.down(w) # [B, K, L]
        
        # 2. 计算记忆检索 (平滑检索)
        # 不使用剧烈的 Softmax，可以尝试更平滑的 Scaling
        logits = torch.matmul(z, self.mb.T) 
        attn = F.softmax(logits / (self.latent_dim ** 0.5), dim=-1)
        
        # 3. 提取语义补丁
        patch = torch.matmul(attn, self.mb) # [B, K, L]
        
        # 4. 映射回原空间并使用 gamma 门控控制
        # 只有在模型确实发现原始特征不足以对齐时，才会通过 gamma 引入补丁
        w_enhanced = w + self.gamma * self.up(patch)
        
        return w_enhanced

class expalignet(nn.Module):
    def __init__(self, text_embed_dim, size = "tiny", pretrained = None, num_classes=80, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.text_embed_dim = text_embed_dim

        # backbone
        self.backbone = ConvNeXt(size=size, pretrained=pretrained)

        backbone_dims = self.backbone.get_embed_dims()[1:]  # 获取 backbone 输出的后三个特征维度列表
        
        # neck
        # self.fusionor = MultiScaleViTFusion(embed_dim=backbone_dims[0], proj_dim=backbone_dims)
        self.neck = FPANeck(backbone_channels=backbone_dims)
        # self.neck = ViTNeck(embed_dim=backbone_dims[0], proj_dim=backbone_dims)
        

        # head（Detect 模块）- 使用 neck 的输出通道
        # neck_out_channels = self.neck.get_output_channels()  # 获取 neck 输出的通道数列表
        self.head = Detect(
            nc=num_classes,
            ch=backbone_dims,
            text_embed_dim=text_embed_dim
        )
        
        # loss（只在训练时使用）
        self.loss_fn = DetectionLoss(
            nc=num_classes,
            reg_max=reg_max,
            stride=self.head.stride,  # 通常来自 head
            device=next(self.parameters()).device
        )
        
        # 初始化检测头偏置
        self.head.bias_init()

        if text_embed_dim > 0:
            self.textffn = nn.Identity()#Residual(SwiGLUFFN(text_embed_dim, text_embed_dim))#

    def forward(self, x, w, m=None, batch=None):
        """
        统一入口：
        - 推理: forward(x) → preds
        - 训练: forward(x, batch) → (loss, loss_items)
        """
        features = self.backbone(x)[1:]      # List[Tensor]，通常 3 个尺度
        neck_out = self.neck(features, w)       # List[Tensor]，融合后的 3 个特征图
        if w is not None and self.text_embed_dim > 0:
            w = self.textffn(w)               # 对文本特征进行投影
        preds = self.head(neck_out[:3], w, m=m)          # List[Tensor] 或 Tensor
        if batch is not None:
            total_loss, loss_items = self.loss_fn(preds, batch)
            return preds, total_loss, loss_items

        return preds, None, None
    
    def set_class(self, text_feats):
        """
        在推理时动态设置类别文本特征（如果使用了文本特征）
        """
        if self.text_embed_dim > 0:
            with torch.no_grad():
                self.head.set_class(text_feats)
                self.loss_fn.update_class_no(text_feats.shape[0])  # 同步更新 loss 中的类别数
    
    def unset_class(self):
        """
        恢复 set_class 之前的模型状态。
        """
        if self.text_embed_dim > 0:
            self.head.unset_class()
            # 恢复 loss_fn 中的类别数
            if hasattr(self.head, '_orig_nc'):
                self.loss_fn.update_class_no(self.head._orig_nc)

