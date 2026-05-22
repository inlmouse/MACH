import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastivesim_multi_positive_infonce(
        sim: torch.Tensor,
        text_is_positive: torch.Tensor,
        temperature: float = 0.07,
        topk_ratio: float = 0.01
    ):
        """
        Vectorized multi-positive contrastive loss.
        sim: [B,K,H,W]
        text_is_positive: [B,K] bool
        Returns scalar loss (or per-image if reduction == 'none')
        """
        device = sim.device
        B, K, H, W = sim.shape
        assert text_is_positive.shape == (B, K)
        sim = sim / (sim.abs().max() + 1e-12)  # 近似归一化

        # 1) pool to [B,K]
        flat = sim.view(B, K, -1)  # [B,K,HW]
        topk_vals, _ = flat.topk(max(1, int(topk_ratio * H * W)), dim=-1)  # [B,K,k]
        logits = topk_vals.mean(dim=-1)      # [B,K]
        #logits = pool_sim(sim, pooling=pooling, topk=topk)
        logits = logits / (temperature + 1e-12)

        # 2) log-softmax across K for each image
        logprob = F.log_softmax(logits, dim=1)  # [B,K]

        # 3) targets: uniform across positives per image
        pos_mask = text_is_positive.to(dtype=logprob.dtype)  # [B,K]
        pos_counts = pos_mask.sum(dim=1, keepdim=True)  # [B,1]

        valid = (pos_counts.squeeze(1) > 0)

        pos_counts_clamped = pos_counts.clamp(min=1.0)
        T = pos_mask / pos_counts_clamped  # [B,K], zero row if no positives

        per_image_loss = - (T * logprob).sum(dim=1)  # [B]
        per_image_loss = per_image_loss[valid]
        if per_image_loss.numel() == 0:
            return torch.tensor(0.0, device=device)
        return per_image_loss.mean()


def sim_patch_level_grpo_objective(sim, M, text_pos_tensor, beta=1.0, gamma=1.0, adv_clip=5.0, eps=1e-6):
        """
        Geometry-Aware Consistency Objective for patch-level masks.
        sim: [B,K,H,W], similarity values in [-1,1]
        M: [B,K,H,W], binary mask 0/1
        text_pos_tensor: [B,K], stack from text_is_positve, bool type
        """
        B, K, H, W = sim.shape
        device = sim.device
        #Npix = H * W
        sim = sim / (sim.abs().max() + 1e-12)
        
        # flatten for softmax
        logits = sim.view(B, -1)      # [B, K*H*W]
        logp = F.log_softmax(logits, dim=1) # [B, K*H*W]
        #p = logp.exp()

        # flatten mask and prob
        M_flat = M.view(B, -1)                   # [B, K*H*W]
        prob_flat = torch.sigmoid(sim).view(B, -1)

        # --- advantage-weighted loss (only positives)
        L_adv_num = torch.tensor(0.0, device=device)
        denom = 0.0
        for b in range(B):
            pos_idx = (M_flat[b] > 0.5).nonzero(as_tuple=True)[0]
            if pos_idx.numel() == 0:
                continue
            R_pos = prob_flat[b, pos_idx]
            mu = R_pos.mean()
            std = R_pos.std(unbiased=False) + eps
            A = (R_pos - mu) / std
            A = A.clamp(-adv_clip, adv_clip)
            logp_pos = logp[b, pos_idx]
            L_adv_num = L_adv_num - (A * logp_pos).sum()
            denom += pos_idx.numel()

        L_adv = (L_adv_num / denom) if denom > 0 else torch.tensor(0.0, device=device)

        # --- negative suppression (BCE)
        neg_mask = ~text_pos_tensor   # [B, K], bool
        #neg_mask = (M.sum(dim=2).sum(dim=2) == 0)  # [B,K], True for mask all 0
        num_neg = neg_mask.sum().clamp(min=1.0)
        if False:# neg_mask.sum() > 0:
            sim_flat = sim.view(B, K, -1)  # [B, K, Npix]
            # target per-pixel for negatives is 0
            target_per_pixel = torch.zeros_like(sim_flat, dtype=sim_flat.dtype, device=sim_flat.device)

            # bce per pixel (no reduction)
            bce_per_pixel = F.binary_cross_entropy_with_logits(sim_flat, target_per_pixel, reduction='none')  # [B,K,Npix]

            # 2) mean over pixels -> mean BCE per (b,k)
            bce_mean_per_k = bce_per_pixel.mean(dim=2)  # [B,K]

            # 3) apply neg_mask (only negatives contribute)
            neg_mask_float = neg_mask.to(sim.device).float()  # [B,K]
            num_neg = neg_mask_float.sum().clamp(min=1.0)

            L_neg = (bce_mean_per_k * neg_mask_float).sum() / num_neg
        else:
            L_neg = torch.tensor(0.0, device=device)

        total_loss = beta * L_adv + gamma * L_neg
        return total_loss#, {'adv': L_adv.detach(), 'neg': L_neg.detach(), 'total': total_loss.detach()}





def multi_positive_contrastive_ranking_loss(
    sim: torch.Tensor,
    batch: dict | None = None,
    temperature: float = 0.07,
    topk_ratio: float = 0.01,
    margin_box: float = 0.3,
    box_topk_ratio: float = 0.01,
    lambda_box: float = 1.0,
):
    """
    Multi-positive contrastive loss + bbox in/out auxiliary loss.

    Args:
        sim: [B, K, H, W]
            fused simmap, e.g. [B, L, 80, 80]
        batch:
            {
                'batch_idx': [N]
                'cls':       [N, 1] or [N]
                'bboxes':    [N, 4] normalized xyxy
                'text_is_positive': [B, K] bool, True 表示该 image 中该 text 是正样本
            }
            若提供，则额外计算 bbox 内 > bbox 外 的损失
        temperature: InfoNCE temperature
        topk_ratio:   取空间 topk 做 pooling 的比例
        margin_box:   bbox 内外约束 margin
        box_topk_ratio: bbox 内外 pooling 的 topk 比例
        lambda_box:   bbox loss 权重

    Returns:
        total_loss, loss_items
    """
    device = sim.device
    B, K, H, W = sim.shape
    text_is_positive = batch["text_is_positive"]

    # ----------------------------------------------------
    # 1) 文本对比损失：正样本 logprob 要高于负样本
    # ----------------------------------------------------
    # 建议按 image 内归一化，比全局 abs.max 更稳一点
    sim = sim / (sim.detach().abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-12)

    flat = sim.view(B, K, -1)  # [B, K, HW]
    k = max(1, int(topk_ratio * H * W))
    topk_vals = flat.topk(k, dim=-1).values  # [B, K, k]
    logits = topk_vals.mean(dim=-1) / (temperature + 1e-12)  # [B, K]

    logprob = F.log_softmax(logits, dim=1)  # [B, K]

    pos_mask = text_is_positive.to(dtype=logprob.dtype)  # [B, K]
    pos_counts = pos_mask.sum(dim=1, keepdim=True)       # [B, 1]
    valid = (pos_counts.squeeze(1) > 0)

    pos_counts_clamped = pos_counts.clamp(min=1.0)
    target = pos_mask / pos_counts_clamped               # [B, K]
    per_image_text_loss = -(target * logprob).sum(dim=1)  # [B]
    per_image_text_loss = per_image_text_loss[valid]

    if per_image_text_loss.numel() == 0:
        loss_text = torch.tensor(0.0, device=device)
    else:
        loss_text = per_image_text_loss.mean()

    # ----------------------------------------------------
    # 2) bbox 内平均 > bbox 外平均  (简化 + 向量化版本)
    # ----------------------------------------------------
    loss_box = torch.tensor(0.0, device=device)
    
    if batch is not None:
        batch_idx = batch["batch_idx"].long().to(device)
        cls_idx   = batch["cls"].view(-1).long().to(device)   # text index
        bboxes    = batch["bboxes"].float().to(device)        # [N, 4] xyxy normalized

        N = batch_idx.shape[0]

        # 过滤无效索引
        valid = (batch_idx >= 0) & (batch_idx < B) & (cls_idx >= 0) & (cls_idx < K)
        if not valid.any():
            loss_box = torch.tensor(0.0, device=device)
        else:
            batch_idx = batch_idx[valid]
            cls_idx   = cls_idx[valid]
            bboxes    = bboxes[valid]

            # 提取对应 sim_map: [N_valid, H, W]
            sim_maps = sim[batch_idx, cls_idx]                     # [N_valid, H, W]

            # 计算 bbox 坐标（像素级）
            x1 = torch.floor(bboxes[:, 0] * W).long()
            y1 = torch.floor(bboxes[:, 1] * H).long()
            x2 = torch.ceil(bboxes[:, 2] * W).long()
            y2 = torch.ceil(bboxes[:, 3] * H).long()

            # 边界处理，保证至少 1 个像素
            x1 = torch.clamp(x1, 0, W - 1)
            y1 = torch.clamp(y1, 0, H - 1)
            x2 = torch.clamp(x2, 0, W)
            y2 = torch.clamp(y2, 0, H)

            # 保证 box 至少 1 个像素
            x2 = torch.maximum(x2, x1 + 1)
            y2 = torch.maximum(y2, y1 + 1)

            n_valid = x1.shape[0]

            # 构建 mask_in [N_valid, H, W]
            mask_in = torch.zeros((n_valid, H, W), dtype=torch.bool, device=device)
            for i in range(n_valid):
                mask_in[i, y1[i]:y2[i], x1[i]:x2[i]] = True

            mask_out = ~mask_in

            # 计算平均值（masked mean）
            def masked_mean(x, mask):
                """x: [N, H, W], mask: [N, H, W] bool"""
                masked_x = x * mask
                sum_val = masked_x.sum(dim=(1, 2))
                count   = mask.sum(dim=(1, 2)).clamp(min=1.0)
                return sum_val / count

            score_in  = masked_mean(sim_maps, mask_in)
            score_out = masked_mean(sim_maps, mask_out)

            # 希望 score_in > score_out + margin_box
            box_loss = F.relu(margin_box - score_in + score_out)   # [N_valid]

            # 取有效 box 的平均
            valid_box = (score_in.isfinite() & score_out.isfinite() & (mask_in.sum(dim=(1,2)) > 0))
            if valid_box.any():
                loss_box = box_loss[valid_box].mean()
            else:
                loss_box = torch.tensor(0.0, device=device)

    # ----------------------------------------------------
    # 3) 总损失
    # ----------------------------------------------------
    total_loss = loss_text + lambda_box * loss_box

    loss_items = {
        "loss_text": loss_text.detach(),
        "loss_box": loss_box.detach(),
        "loss_total": total_loss.detach(),
    }

    return total_loss, loss_items