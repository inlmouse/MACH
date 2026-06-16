import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.detect_utils import make_anchors, dist2bbox, bbox_iou, bbox2dist, xywh2xyxy
from loss.taskalignedassigner import TaskAlignedAssigner
from loss.auxloss import multi_positive_contrastive_ranking_loss



class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

class DetectionLoss(nn.Module):
    """
    Detection Loss
    - 继承 nn.Module，使用 forward 方法
    - 显式传入必要参数，不依赖整个 model
    - 适用于普通训练和推理场景
    """
    def __init__(
        self,
        nc: int = 80,                     # 类别数
        reg_max: int = 16,                # DFL 的 bin 数量
        stride: list = [8, 16, 32],       # 三个尺度的 stride
        hyp=None,                         # 超参 (box, cls, dfl 等权重)
        tal_topk: int = 10,               # TaskAlignedAssigner 的 topk
        device: torch.device = None,
    ):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.as_tensor(stride, dtype=torch.float32).detach()
        self.hyp = hyp or type('Hyp', (), {'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'jepa': 0.1})()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.use_dfl = reg_max > 1
        self.tal_topk = tal_topk

        # Assign & Loss components
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(reg_max).to(self.device)
        self.proj = torch.arange(reg_max, dtype=torch.float, device=self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """预处理 targets，将其按 batch 分组，并转为 xyxy 格式"""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """从分布预测解码出 bbox 坐标"""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            # 确保 proj 与 pred_dist 在同一设备上
            proj = self.proj.to(pred_dist.device)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, preds, batch, jepa_loss=None):
        """
        preds: List[Tensor] 或 Tuple，来自 neck 的多尺度特征图
        batch: dict，包含 'batch_idx', 'cls', 'bboxes' 等键
        返回: (total_loss, [box_loss, cls_loss, dfl_loss])
        """
        # 从 preds 动态获取设备（模型可能被移动到不同设备）
        device = preds[0].device if isinstance(preds, (list, tuple)) else preds[1][0].device
        self.device = device
        loss = torch.zeros(4, device=device)  # [box, cls, dfl, jepa]
        if jepa_loss is not None and self.training:
            loss[3] = jepa_loss * self.hyp.jepa

        # 处理 preds（多尺度特征图）
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], dim=2
        ).split((self.reg_max * 4, self.nc), dim=1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # B, num_anchors, nc
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # B, num_anchors, reg_max*4

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0].to(device)

        # 生成 anchors 和 stride tensor
        anchor_points, stride_tensor = make_anchors(feats, self.stride.to(device), 0.5)

        # 处理 targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), dim=1
        )
        targets = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(dim=2, keepdim=True).gt_(0.0)

        # 预测的 bbox（用于 assigner）
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Task-Aligned Assign
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1.0)

        # 分类损失 (BCE)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # 边界框 & DFL 损失
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask
            )

        # 应用超参权重
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        # 总损失 × batch_size（保持与原版一致，便于多卡平均）
        total_loss = loss.sum() * batch_size
        loss_items = {
            "loss_bbox": loss[0].detach().item(),
            "loss_class": loss[1].detach().item(),
            "loss_dfl": loss[2].detach().item(),
            "loss_jepa": loss[3].detach().item() if jepa_loss is not None else 0.0,
            "total_loss": total_loss.detach().item() / max(batch_size, 1) # 返回平均值供监控
        }
        return total_loss, loss_items  # 返回 scalar loss 和 [box, cls, dfl] 用于日志

    def update_class_no(self, new_nc):
        """动态更新类别数（如 fine-tune 时）"""
        self.nc = new_nc
        self.no = self.nc + self.reg_max * 4
        self.assigner = TaskAlignedAssigner(topk=self.tal_topk, num_classes=new_nc, alpha=0.5, beta=6.0)


# 使用示例
if __name__ == "__main__":
    # 假设你有这些
    loss_fn = DetectionLoss(nc=80, reg_max=16, stride=[8, 16, 32])
    
    # 模拟输入
    preds = [torch.randn(4, 84, 80, 80), torch.randn(4, 84, 40, 40), torch.randn(4, 84, 20, 20)]
    batch = {
        "batch_idx": torch.tensor([0,0,1,1,2,2,3]),
        "cls": torch.randn(7, 1),
        "bboxes": torch.randn(7, 4),
    }
    
    total_loss, loss_items = loss_fn(preds, batch)
    print("Total loss:", total_loss.item())
    print("Box | Cls | DFL:", loss_items.tolist())