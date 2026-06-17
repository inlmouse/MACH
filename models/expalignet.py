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
        neck_out = self.neck(features, w, m, batch)       # List[Tensor]，融合后的 3 个特征图
        if w is not None and self.text_embed_dim > 0:
            w = self.textffn(w)               # 对文本特征进行投影
        preds, jepa_loss = self.head(neck_out[:3], w, m=m, batch=batch)          # List[Tensor] 或 Tensor
        if self.training:
            # 防御性断言：既然是训练模式，调用方必须传 batch 标签！
            if batch is None:
                raise ValueError("处于训练模式 (self.training=True) 时，必须传入 batch 参数计算 Loss！")
                
            total_loss, loss_items = self.loss_fn(preds, batch, jepa_loss)
            return preds, total_loss, loss_items

        return preds, None, None
    
    @torch.no_grad()
    def predict(self, x, w, m=None, conf_threshold=0.25, nms_threshold=0.45, orig_target_sizes=None):
        """
        端到端推理接口（YOLO 架构兼容版）。
        内置动态类别切换、NMS 过滤及坐标还原，输出格式与 vlrtdetrnet 保持一致。
        
        Args:
            x (torch.Tensor): 图像张量 [B, 3, H, W]，已归一化。
            w (torch.Tensor): 文本特征 [B, nc, L, Dim] 或 [B, nc, Dim]。
            m (torch.Tensor, optional): 文本 Mask。
            conf_threshold (float): 置信度过滤阈值。
            nms_threshold (float): NMS IoU 阈值。
            orig_target_sizes (torch.Tensor, optional): 原始图像尺寸 [B, 2]，格式为 (高, 宽)。
        
        Returns:
            List[Dict]: 长度为 B 的列表，包含 boxes (绝对坐标 xyxy), scores, labels
        """
        was_training = self.training
        self.eval()

        # 1. 挂载动态文本特征 (改变模型内部卷积/线性层权重)
        self.set_class(w)

        try:
            # 2. 前向传播
            # 即使 head 缓存了 class，我们依然按照签名传入 w，让 neck 或其他组件正常工作
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                forward_outs = self.forward(x, w, m=m)
            preds = forward_outs[0] if isinstance(forward_outs, tuple) else forward_outs
            
        finally:
            # 3. 护城河级防御：无论推理成功与否，必定卸载特征，防止内存泄漏和串图
            self.unset_class()
            if was_training:
                self.train()

        # 4. 后处理: 引入你的 NMS 与 坐标缩放工具
        from utils.detect_utils import non_max_suppression, scale_boxes
        
        # 执行 NMS (输出列表，长度为 B，每个元素是 [num_dets, 6] 的 Tensor)
        preds_after_nms = non_max_suppression(preds, conf_thres=conf_threshold, iou_thres=nms_threshold)

        results = []
        B = x.shape[0]
        input_h, input_w = x.shape[2], x.shape[3]

        # 5. 逐图解析与坐标还原
        for i in range(B):
            pred_i = preds_after_nms[i]

            # 处理该图片没有检测到任何目标的情况
            if pred_i is None or len(pred_i) == 0:
                results.append({
                    "boxes": torch.empty((0, 4), device=x.device),
                    "scores": torch.empty((0,), device=x.device),
                    "labels": torch.empty((0,), device=x.device, dtype=torch.long)
                })
                continue

            # NMS 输出的通常是基于输入图像尺寸 (如 640x640) 的绝对坐标 [x1, y1, x2, y2]
            final_boxes = pred_i[:, :4]
            final_scores = pred_i[:, 4]
            final_labels = pred_i[:, 5].long()

            # 如果传入了原图尺寸，将框从 640x640 映射回真实分辨率
            if orig_target_sizes is not None:
                orig_h, orig_w = orig_target_sizes[i]
                final_boxes = scale_boxes(
                    img1_shape=(input_h, input_w),
                    boxes=final_boxes,
                    img0_shape=(orig_h, orig_w)
                )

            results.append({
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_labels
            })

        return results
    
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

