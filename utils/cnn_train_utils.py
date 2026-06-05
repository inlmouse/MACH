# utils/train_utils.py
# 训练工具函数，兼容单卡和 DDP 训练

import os
from pathlib import Path
import math
import torch
import torch.distributed as dist
from torch.amp import autocast
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

import wandb
import datetime
#from test import inference_single, draw_predictions

def setup_distributed():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            device_id=torch.device(f'cuda:{local_rank}'),
            timeout=datetime.timedelta(seconds=1200)
        )
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0

def load_model(ckpt_path, text_embed_dim, num_classes=None, device='cuda'):
    """加载训练好的模型，自动推断类别数"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    epoch = ckpt.get('epoch', 0) + 1
    state_dict = ckpt['model_state_dict']
    optimizer_state_dict = ckpt['optimizer_state_dict']
    if num_classes is None:
        # 从 head 的权重推断类别数
        for key, val in state_dict.items():
            if 'head.cv3' in key and '.12.weight' in key and len(val.shape) == 4:
                num_classes = val.shape[0]
                print(f"从模型权重推断类别数: {num_classes}")
                break
        else:
            raise ValueError("无法从模型权重推断类别数，请检查 checkpoint 格式")
    from models.expalignet import expalignet
    model = expalignet(
        text_embed_dim = text_embed_dim,
        size="tiny",
        pretrained=None,
        num_classes=num_classes
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    base_lr = 0.002
    weight_decay = 0.025
    # 创建 optimizer - 直接使用所有参数，不硬编码 param groups
    # optimizer state_dict 会正确恢复每个参数的状态和学习率
    AdamW_optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': base_lr},
            {'params': model.neck.parameters(), 'lr': base_lr},
            {'params': [p for n, p in model.head.named_parameters() if 'alignhead' not in n.lower()], 'lr': base_lr},
            {'params': model.head.alignhead.parameters(), 'lr': base_lr * 1},     # BNContrastiveHead
        ], weight_decay=weight_decay)
    AdamW_optimizer.load_state_dict(optimizer_state_dict)

    return model, AdamW_optimizer, epoch

def get_scheduler(optimizer, warmup_epochs: float, num_epochs: int, start_epoch: int = 0):
    """
    全局 Cosine Annealing + 前期 Linear Warmup 线性逼近
    """
    def lr_lambda(current_epoch: float):
        # current_epoch 支持小数（如 warmup_epochs=1.5）
        progress = current_epoch / num_epochs   # 全局进度 [0, 1]
        lrf = 0.01  # 最终学习率相对于 base_lr 的比例（cosine 下降到这个比例）
        
        # 全局 cosine 值（从 1.0 逐渐下降到接近 0）
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress)) * (1 - lrf) + lrf
        
        if current_epoch < warmup_epochs:
            # 线性逼近：把全局 cosine 在当前 epoch 的值作为目标，线性上升到它
            target = cosine_factor
            return (current_epoch + 1.0) / warmup_epochs * target
        else:
            # 直接使用全局 cosine
            return cosine_factor

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # resume 支持
    for _ in range(start_epoch - 1):
        scheduler.step()

    return scheduler

def validate_one_epoch(
    model,
    dataloader,
    device,
    epoch,
    txt_feats,
    label_list,
    target_size,
    compute_loss=True,
    is_main_process=True,
    textencoder=None,
    output_dir=None,
    visualize_img_dict=
    {
        "/project/GLS/HJY/VLMs/test_results/test2.jpg": ['loafers', 'knee-high socks', 'pleated skirt', 'bag', 'sailor uniform', 'ponytail'],
        "/data/OOD/coco/images/val2017/000000367569.jpg": ['bottle', 'wall painting', 'table', 'fireplace', 'stool', 'curtain', 'laptop', 'remote control', 'lamp', 'book', 'lamp on ceiling'],
    }
):
    """
    验证一个 epoch，计算 mAP 和（可选）损失。
    支持单卡和 DDP 训练。
    
    Args:
        model: 训练模型（DDP 包装或普通模型）
        dataloader: 验证数据加载器
        device: 计算设备
        epoch: 当前 epoch 数
        txt_feats: 文本编码特征（已预计算并传入）
        label_list: 类别列表
        compute_loss: 是否计算验证损失
        is_main_process: 是否为主进程（DDP 场景）
    
    Returns:
        dict: {
            'map': mAP@0.5:0.95,
            'map_50': mAP@0.5,
            'loss': 平均损失（如果 compute_loss=True）
        }
    """
    from evaluation.coco_evaluator import DetectionEvaluator, print_map_results
    from utils.detect_utils import non_max_suppression
    
    # 只在主进程上进行验证（DDP 场景）
    if not is_main_process:
        return {'map': 0.0, 'map_50': 0.0, 'loss': 0.0}
    
    
    # 准备验证模型
    val_model = model.module if hasattr(model, 'module') else model
    val_model.eval()        
    
    # 设置类别文本特征
    all_labels = []
    for prompts in label_list:
        if isinstance(prompts, (list, tuple)):
            all_labels.extend(prompts)
        else:
            all_labels.append(prompts)

    val_model.set_class(txt_feats)
    
    # 初始化评估器
    evaluator = DetectionEvaluator(num_classes=len(all_labels), fast_mode=False)
    
    total_loss = 0.0
    total_loss_items = torch.zeros(4, device=device) if compute_loss else None
    
    pbar = tqdm(dataloader, desc=f"Validate {epoch}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            imgs = batch['img'].to(device)
            batch_size = imgs.shape[0]
            
            batch_data = {
                'batch_idx': batch['batch_idx'].to(device),
                'cls': batch['cls'].to(device),
                'bboxes': batch['bboxes'].to(device),
            }
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # 单次前向传播，同时计算损失和预测
                # 假设模型支持同时接收 txt_feats 和 batch_data
                outputs = val_model(imgs, txt_feats, batch_data)
                
                # 解包结果：模型需要同时返回 pred 和 loss
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    pred, loss, loss_items = outputs
                else:
                    pred = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = None
                    loss_items = None
            
            # 累加损失
            if compute_loss and loss is not None:
                loss_per_img = loss.item() / batch_size
                total_loss += loss_per_img
                total_loss_items += loss_items.to(device)
            
            # NMS
            preds = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.65, max_det=300)
            
            # 处理 GT 和预测结果
            gt_batch_idx = batch['batch_idx'].to(device)
            gt_cls = batch['cls'].to(device)
            gt_bboxes = batch['bboxes'].to(device)
            
            # 收集 batch 中每张图的结果
            batch_pred_boxes = []
            batch_pred_scores = []
            batch_pred_labels = []
            batch_gt_boxes = []
            batch_gt_labels = []
            batch_img_ids = []
            batch_gt_areas = []
            
            for i in range(batch_size):
                img_id = batch_idx * batch_size + i
                
                # 预测结果
                pred_i = preds[i]
                if pred_i is not None and len(pred_i) > 0:
                    pred_boxes = pred_i[:, :4]  # xyxy
                    pred_scores = pred_i[:, 4]   # conf
                    pred_labels = pred_i[:, 5].long()  # cls
                else:
                    pred_boxes = torch.empty((0, 4), device=device)
                    pred_scores = torch.empty((0,), device=device)
                    pred_labels = torch.empty((0,), dtype=torch.long, device=device)
                
                # GT
                mask = gt_batch_idx == i
                if mask.sum() > 0:
                    gt_boxes = gt_bboxes[mask]  # cxcywh
                    gt_labels = gt_cls[mask].squeeze(-1).long()
                    
                    # cxcywh (normalized) -> xyxy (absolute, target_size x target_size)
                    cx, cy, w, h = gt_boxes.T
                    x1 = (cx - w/2) * target_size
                    y1 = (cy - h/2) * target_size
                    x2 = (cx + w/2) * target_size
                    y2 = (cy + h/2) * target_size
                    gt_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                else:
                    gt_boxes = torch.empty((0, 4), device=device)
                    gt_labels = torch.empty((0,), dtype=torch.long, device=device)
                    gt_areas = torch.empty((0,), device=device)
                
                batch_pred_boxes.append(pred_boxes)
                batch_pred_scores.append(pred_scores)
                batch_pred_labels.append(pred_labels)
                batch_gt_boxes.append(gt_boxes)
                batch_gt_labels.append(gt_labels)
                batch_img_ids.append(img_id)
                batch_gt_areas.append(gt_areas)
            
            # 使用 add_batch 一次性添加 batch 结果
            evaluator.add_batch(
                batch_pred_boxes,
                batch_pred_scores,
                batch_pred_labels,
                batch_gt_boxes,
                batch_gt_labels,
                batch_img_ids,
                batch_gt_areas
            )
            
            # 更新进度条
            if compute_loss:
                num_batches = pbar.n + 1
                avg_loss_items = (total_loss_items / num_batches).cpu()
                pbar.set_postfix({
                    'box': f"{avg_loss_items[0]:.4f}",
                    'cls': f"{avg_loss_items[1]:.4f}",
                    'dfl': f"{avg_loss_items[2]:.4f}"
                })
    
    """if output_dir is not None and visualize_img_dict is not None:
        vis_test_feat = textencoder.embedtext(class_names, normalize=True)  # 可视化时也需要设置文本特征
        val_model.set_class(vis_test_feat)
        for img_path, class_names in visualize_img_dict.items():
            image, boxes, scores, labels = inference_single(
                model, str(img_path), device, 640, 
                0.1, 0.5
            )
            vis_image = draw_predictions(
                image, boxes, scores, labels, 
                class_names=class_names,
            )
            save_name = Path(img_path).stem + f'_result_epoch{epoch}.jpg'
            save_path = os.path.join(output_dir, save_name)
            vis_image.save(save_path)"""

    # 恢复模型到 set_class 之前的状态
    val_model.unset_class()

    # 计算 mAP
    map_results = evaluator.compute_map(conf_thresh=0.001)
    map_50_95 = map_results.get('mAP@0.5:0.95', 0.0)
    map_50 = map_results.get('mAP@0.5', 0.0)
    
    # 打印结果
    if compute_loss:
        avg_loss = total_loss / len(dataloader)
        avg_loss_items = (total_loss_items / len(dataloader)).cpu()
        print(f"Validate {epoch} average loss - box: {avg_loss_items[0]:.4f}, cls: {avg_loss_items[1]:.4f}, dfl: {avg_loss_items[2]:.4f}, total: {avg_loss:.4f}")
    
    print(f"Validate {epoch} mAP - mAP@0.5:0.95: {map_50_95:.4f}, mAP@0.5: {map_50:.4f}")

    result = {
        'map': map_50_95,
        'map_50': map_50,
    }
    
    if compute_loss:
        result['loss'] = avg_loss
        result['loss_items'] = avg_loss_items.tolist()
    evaluator.reset()
    return result


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    epoch,
    device,
    scheduler=None,
    train_sampler=None,
    use_amp=True,
    is_main_process=True,
    textencoder=None,
    world_size=1,
    wandb_run=None,
    log_interval=10,  # 每多少个 batch 记录一次
    all_caption_embeddings = None,
    caption_to_idx = None,
):
    """
    训练一个 epoch，兼容单卡和 DDP 训练
    
    Args:
        model: 模型（DDP 包装或普通模型）
        dataloader: 训练数据加载器
        optimizer: 优化器
        scaler: GradScaler（混合精度）
        epoch: 当前 epoch
        device: 计算设备
        scheduler: 学习率调度器（可选）
        train_sampler: DistributedSampler（DDP 场景）
        use_amp: 是否使用混合精度
        is_main_process: 是否为主进程
        world_size: 进程数（DDP 场景）
        use_wandb: 是否使用 wandb 记录
        log_interval: 每多少个 batch 记录一次 wandb
    
    Returns:
        dict: {
            'loss': 平均损失,
            'loss_items': [box_loss, cls_loss, dfl_loss],
            'lr': 学习率
        }
    """
    import torch.distributed as dist
    
    # DDP 场景：设置 sampler 的 epoch
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    if all_caption_embeddings is None and caption_to_idx is None:
        tokenlevel_embdedding = True
    # elif (all_caption_embeddings is not None) and (caption_to_idx is not None):
    #     tokenlevel_embdedding = False
    else:
        tokenlevel_embdedding = False
        
    model.train()
    total_loss = 0.0
    total_loss_items = torch.zeros(4, device=device) #
    
    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
        
    # 进度条（仅主进程显示）
    if is_main_process:
        #print(f"Current LR: {current_lr:.6f}")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    # 计算全局 step 起始值
    global_step = (epoch - 1) * len(dataloader) if hasattr(dataloader, '__len__') else 0
    
    for batch_idx, batch in enumerate(pbar):
        imgs = batch['img'].to(device)
        B = imgs.shape[0]
        batch_captions = batch.get('batch_captions', [])
        batch_captions = [s[:300] for s in batch_captions]
        if tokenlevel_embdedding:
            textfeats, mask = textencoder.embedtext(batch_captions, normalize=True, batch_size=len(batch_captions), tokenlevel=tokenlevel_embdedding)
            _, L, C = textfeats.shape
            textfeats = textfeats.to(device)
            mask = mask.to(device)
            # [B*nc, L, C] -> [B, nc, L, C] -> [B, L, nc, C] -> [B, L*nc, C]
            # textfeats = textfeats.view(B, -1, L, C).permute(0, 2, 1, 3).reshape(B, -1, C)
            # mask = mask.view(B, -1, L).permute(0, 2, 1).reshape(B, -1)
        else:
            indices = [caption_to_idx[cap] for cap in batch_captions]
            textfeats = all_caption_embeddings[indices]
            _, C = textfeats.shape
            textfeats = textfeats.view(B, -1, C)
            textfeats = textfeats.to(device)
            mask = None
        
        batch_data = {
            'batch_idx': batch['batch_idx'].to(device),
            'cls': batch['cls'].to(device),
            'bboxes': batch['bboxes'].to(device),
            'text_is_positive' : batch['text_is_positive'].to(device) if 'text_is_positive' in batch else None,
        }
        
        optimizer.zero_grad()
        
        # 前向传播
        with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            _, loss, loss_items = model(imgs, textfeats, mask, batch_data)
        
        # 反向传播
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # 累积损失
        loss_per_img = loss.item() / imgs.shape[0]
        total_loss += loss_per_img
        total_loss_items += loss_items.to(device)
        

        
        # 当前 batch 的损失（用于 wandb）
        batch_loss_items = loss_items.cpu()
        
        # 更新进度条（仅主进程）
        if is_main_process:
            num_batches = pbar.n + 1
            avg_loss_items = (total_loss_items / num_batches).cpu()
            if len(avg_loss_items) >= 3:
                pbar.set_postfix({
                    'box': f"{avg_loss_items[0]:.4f}",
                    'cls': f"{avg_loss_items[1]:.4f}",
                    'dfl': f"{avg_loss_items[2]:.4f}",
                    'jepa': f"{avg_loss_items[3]:.4f}"
                })
            else:
                pbar.set_postfix({
                    'box': f"{avg_loss_items[0]:.4f}",
                    'cls': f"{avg_loss_items[1]:.4f}",
                    'dfl': f"{avg_loss_items[2]:.4f}"
                })
            
            # 记录每个 batch 到 wandb
            if (wandb_run is not None) and (batch_idx % log_interval == 0 or batch_idx == len(dataloader) - 1):
                step = global_step + batch_idx
                wandb_run.log({
                    "batch/loss": loss_per_img,
                    "batch/box_loss": batch_loss_items[0].item(),
                    "batch/cls_loss": batch_loss_items[1].item(),
                    "batch/dfl_loss": batch_loss_items[2].item(),
                    "batch/jepa_loss": batch_loss_items[3].item(),
                    "batch/lr": current_lr,
                }, step=step)
    
    # 更新学习率
    if scheduler is not None:
        scheduler.step()
    
    # DDP 场景：同步所有进程的损失
    if world_size > 1:
        dist.all_reduce(total_loss_items, op=dist.ReduceOp.AVG)
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_loss_items = (total_loss_items / len(dataloader)).cpu()
    
    #if is_main_process:
        #print(f"Epoch {epoch} 平均损失 - box: {avg_loss_items[0]:.4f}, cls: {avg_loss_items[1]:.4f}, dfl: {avg_loss_items[2]:.4f}, total: {avg_loss:.4f}")
    
    return {
        'loss': avg_loss,
        'loss_items': avg_loss_items.tolist(),
        'lr': current_lr,
    }


def save_checkpoint(model, optimizer, epoch, output_dir, is_best=False, is_main_process=True):
    """
    保存模型检查点
    
    Args:
        model: 模型（DDP 包装或普通模型）
        optimizer: 优化器
        epoch: 当前 epoch
        output_dir: 输出目录
        is_best: 是否为最佳模型
        is_main_process: 是否为主进程
    """
    if not is_main_process:
        return
    
    # 获取模型状态（处理 DDP 包装）
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if is_best:
        path = os.path.join(output_dir, "model_best.pth")
        torch.save(checkpoint, path)
    
    path = os.path.join(output_dir, f"model_last.pth")
    torch.save(checkpoint, path)
    print(f"Saved {epoch}-th epoch checkpoint to {path}")
