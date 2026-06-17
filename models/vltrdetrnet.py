import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.fpn import FPANeck
from layers.vitneck import ViTNeck
from layers.vldetrhead import VLRTDETRDecoder
from loss.dertloss import RTDETRDetectionLoss
from models.convnext import ConvNeXt


class vlrtdetrnet(nn.Module):
    def __init__(self, text_embed_dim, size = "tiny", pretrained = None, num_classes=80, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.text_embed_dim = text_embed_dim

        # backbone
        self.backbone = ConvNeXt(size=size, pretrained=pretrained)

        backbone_dims = self.backbone.get_embed_dims()[1:]  # 获取 backbone 输出的后三个特征维度列表
        
        # neck
        self.neck = FPANeck(backbone_channels=backbone_dims)
        # self.neck = ViTNeck(proj_dim=backbone_dims)

        # head（Detect 模块）- 使用 neck 的输出通道
        # neck_out_channels = self.neck.get_output_channels()  # 获取 neck 输出的通道数列表
        self.head = VLRTDETRDecoder(
            nc=num_classes,
            ch=backbone_dims,
            hd=text_embed_dim
        )
        
        # loss（只在训练时使用）
        self.loss_fn = RTDETRDetectionLoss(
            nc=num_classes,
        )
        

    def forward(self, x, w, m=None, batch=None):
        """
        统一入口：
        - 推理 (self.training=False): 返回预测结果 preds
        - 训练 (self.training=True): 返回 preds, total_loss (标量), loss_dict (用于日志记录的字典)
        """
        # 1. 骨干网络与多尺度特征融合
        features = self.backbone(x)[1:]      
        neck_out = self.neck(features, w, m, batch)  

        # ==========================================
        # 训练模式
        # ==========================================
        if self.training:
            # 假设你的定制 head 返回了 解码器大元组 和 额外的表征对齐损失(jepa_loss)
            head_outs = self.head(neck_out[:3], w, m=m, batch=batch)
            
            # 解析 Decoder 的原生输出
            # dec_bboxes/scores 包含了 [num_layers, B, num_dn + num_queries, ...]
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = head_outs
            
            dn_bboxes, dn_scores = None, None
            pred_bboxes, pred_scores = dec_bboxes, dec_scores
            
            # 核心逻辑：拆分去噪 (DN) 结果与普通预测结果
            if dn_meta is not None:
                # dn_num_split 记录了 [去噪Query数量, 普通Query数量]
                num_dn, num_queries = dn_meta["dn_num_split"]
                
                # 在序列维度 (dim=-2) 上进行切分
                # 切分后: 
                # dn_bboxes: [num_layers, B, num_dn, 4]
                # pred_bboxes: [num_layers, B, num_queries, 4]
                dn_bboxes, pred_bboxes = torch.split(dec_bboxes, [num_dn, num_queries], dim=-2)
                dn_scores, pred_scores = torch.split(dec_scores, [num_dn, num_queries], dim=-2)
            
            # 组装目标检测 Loss 需要的 preds
            preds = (pred_bboxes, pred_scores)
            
            # 计算检测与去噪 Loss
            # 注意：你的 RTDETRDetectionLoss 返回的是一个字典 dict[str, torch.Tensor]
            loss_dict = self.loss_fn(
                preds=preds, 
                batch=batch, 
                dn_bboxes=dn_bboxes, 
                dn_scores=dn_scores, 
                dn_meta=dn_meta,
                enc_bboxes=enc_bboxes,  # two stage supervision
                enc_scores=enc_scores   # two stage supervision
            )
            
            # 聚合总 Loss
            # ['loss_class', 'loss_bbox', 'loss_giou', 'loss_class_aux', 'loss_bbox_aux', 'loss_giou_aux', 'loss_class_dn', 'loss_bbox_dn', 'loss_giou_dn', 'loss_class_aux_dn', 'loss_bbox_aux_dn', 'loss_giou_aux_dn']
            total_loss = sum(loss_dict.values())
            
            return preds, total_loss, loss_dict

        # ==========================================
        # 推理模式
        # ==========================================
        else:
            # 推理时，Decoder 通常会直接返回经过 NMS/Top-K 后处理的最终张量
            preds = self.head(neck_out[:3], w, m=m, batch=batch)
            return preds, None, None
    
    @torch.no_grad()
    def predict(self, x, w, m=None, conf_threshold=0.5, orig_target_sizes=None):
        """
        端到端推理接口：输入图像和文本特征，输出过滤后的边界框、得分和标签。
        
        Args:
            x (torch.Tensor): 图像张量 [B, 3, H, W]，已归一化。
            w (torch.Tensor): 文本特征 [B, nc, L, Dim] (未池化的文本词向量)。
            m (torch.Tensor, optional): 文本 Mask [B, nc, L] (1为有效, 0为Padding)。
            conf_threshold (float): 置信度阈值，过滤掉低分预测。
            orig_target_sizes (torch.Tensor, optional): 原始图像尺寸 [B, 2]，格式为 (高, 宽) 或 (宽, 高)
                                                        传入后会自动将框还原到原图绝对像素坐标。
        
        Returns:
            List[Dict]: 长度为 B 的列表，每个字典包含单张图的预测结果:
                {
                    "boxes": Tensor [num_dets, 4],  # 边界框
                    "scores": Tensor [num_dets],    # 置信度
                    "labels": Tensor [num_dets]     # 文本短语/类别索引
                }
        """
        # 1. 确保模型处于推理模式
        was_training = self.training
        self.eval()
        
        # 2. 前向传播
        # 在非 training 模式下，forward 返回 (preds, None, None)，preds 
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            forward_outs = self.forward(x, w, m=m)[0]
        preds = forward_outs[0] if isinstance(forward_outs, tuple) else forward_outs
        
        # preds 是 decoder.postprocess 的输出
        # 形状预期为: [B, num_queries, 6] -> 最后一维是 [cx, cy, w, h, max_score, class_index]
        B = preds.shape[0]
        results = []
        
        # 3. 逐图解析与过滤
        for i in range(B):
            pred_i = preds[i]  # [num_queries, 6]
            
            # 分离坐标、得分和标签
            boxes = pred_i[:, :4]        # 归一化的 [cx, cy, w, h]
            scores = pred_i[:, 4]        # 经 sigmoid 后的最高置信度
            labels = pred_i[:, 5].long() # 匹配到的类别/文本短语索引
            
            # 置信度阈值过滤
            keep_idx = scores > conf_threshold
            
            final_boxes = boxes[keep_idx]
            final_scores = scores[keep_idx]
            final_labels = labels[keep_idx]
            
            # 4. 可选：坐标转换 (归一化 cxcywh -> 绝对坐标 xyxy)
            if orig_target_sizes is not None and len(final_boxes) > 0:
                # 假设 orig_target_sizes 为 [H, W]
                orig_h, orig_w = orig_target_sizes[i]
                
                # cxcywh 转换逻辑
                cx, cy, w_box, h_box = final_boxes.unbind(-1)
                x1 = (cx - 0.5 * w_box) * orig_w
                y1 = (cy - 0.5 * h_box) * orig_h
                x2 = (cx + 0.5 * w_box) * orig_w
                y2 = (cy + 0.5 * h_box) * orig_h
                
                final_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                
                # 限制框在图像边界内 (Clamp)
                final_boxes[:, 0::2].clamp_(min=0, max=orig_w)
                final_boxes[:, 1::2].clamp_(min=0, max=orig_h)

            results.append({
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_labels
            })

        if was_training:
            self.train()

        return results
    
    # def set_class(self, text_feats):
    #     """
    #     在推理时动态设置类别文本特征（如果使用了文本特征）
    #     """
    #     if self.text_embed_dim > 0:
    #         with torch.no_grad():
    #             self.head.set_class(text_feats)
    #             self.loss_fn.update_class_no(text_feats.shape[0])  # 同步更新 loss 中的类别数
    
    # def unset_class(self):
    #     """
    #     恢复 set_class 之前的模型状态。
    #     """
    #     if self.text_embed_dim > 0:
    #         self.head.unset_class()
    #         # 恢复 loss_fn 中的类别数
    #         if hasattr(self.head, '_orig_nc'):
    #             self.loss_fn.update_class_no(self.head._orig_nc)

