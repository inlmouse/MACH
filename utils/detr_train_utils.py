# utils/train_utils.py
# 训练工具函数，兼容单卡和 DDP 训练，适配 VL-RT-DETR 架构

import os
from pathlib import Path
import math
import torch
import torch.distributed as dist
from torch.amp import autocast
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from models.builder import build_model

import wandb
import datetime

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

def _build_optimizer(model, base_lr, weight_decay):
    """
    辅助函数：统一处理优化器参数分组与学习率分配
    (将 Backbone 的学习率按惯例设为 base_lr * 0.1)
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=base_lr, 
        weight_decay=weight_decay
    )
    
    return optimizer


def load_model(ckpt_path, args, device='cuda', base_lr=1e-4, weight_decay=1e-4, resume=True):
    """
    利用 Factory 模式加载模型与权重。
    
    Args:
        ckpt_path (str): 权重路径。如果为空/None，则从头构建模型。
        args (dict): 模型构建参数。
        device (str): 运行设备。
        base_lr (float): 基础学习率，作为参数传入。
        weight_decay (float): 权重衰减，作为参数传入。
        resume (bool): 是否为断点续训。
                       True -> 加载 Optimizer 状态并继承 Epoch。
                       False -> 微调模式：丢弃 Optimizer，Epoch 归 0。
    """
    # ==========================================
    # 1. 从头训练 (From Scratch)
    # ==========================================
    if not ckpt_path:
        # print("🌟 ckpt_path 为空，正在从头构建全新模型...")
        if args.get('num_classes') is None:
            args['num_classes'] = 80
            print(f"⚠️ 未指定类别数，使用默认类别数: {args['num_classes']}")
            
        args['device'] = device
        model = build_model(args)
        model.train()
        
        AdamW_optimizer = _build_optimizer(model, base_lr, weight_decay)
        return model, AdamW_optimizer, 1

    # ==========================================
    # 2. 预处理 Checkpoint
    # ==========================================
    # print(f"📂 正在加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt) # 兼容有些直存权重的 pt 文件
    
    # 核心逻辑：控制断点续训 vs 微调
    if resume:
        epoch = ckpt.get('epoch', 1)
        optimizer_state_dict = ckpt.get('optimizer_state_dict', None)
        # print(f"▶️ 续训模式开启：继承历史 Optimizer，从 Epoch {epoch} 继续。")
    else:
        epoch = 1
        optimizer_state_dict = None
        # print("🔄 微调模式开启：仅加载模型权重，Optimizer 丢弃，Epoch 重置为 1。")

    # 动态推断类别数逻辑 (兼容旧版 expalignet)
    if args.get('num_classes') is None:
        for key, val in state_dict.items():
            if 'head.cv3' in key and '.12.weight' in key and len(val.shape) == 4:
                args['num_classes'] = val.shape[0]
                print(f"🔄 从旧版权重推断出类别数: {args['num_classes']}")
                break
        else:
            args['num_classes'] = 80 
            print(f"⚠️ 无法推断类别数，使用默认类别数: {args['num_classes']}")

    # ==========================================
    # 3. 实例化模型并智能加载权重
    # ==========================================
    args['device'] = device
    model = build_model(args)
    
    model_dict = model.state_dict()
    pretrained_dict = {}
    matched_keys, ignored_keys = 0, 0
    
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            pretrained_dict[k] = v
            matched_keys += 1
        else:
            ignored_keys += 1
            
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"📦 权重加载完成：成功匹配 {matched_keys} 层，忽略/重置 {ignored_keys} 层。")
    
    model.train()

    # ==========================================
    # 4. 构建优化器并尝试恢复状态
    # ==========================================
    AdamW_optimizer = _build_optimizer(model, base_lr, weight_decay)
    
    if optimizer_state_dict is not None:
        try:
            AdamW_optimizer.load_state_dict(optimizer_state_dict)
            print("✅ 优化器状态加载成功！")
        except ValueError as e:
            print("⚠️ 架构或参数尺寸已变，旧的 Optimizer 状态不兼容，已自动重置动量。")

    return model, AdamW_optimizer, epoch

def get_scheduler(optimizer, warmup_steps: int, epochs: int, steps_per_epoch: int):
    """
    DINO 风格调度器 (Step-based)：
    - 前期：按 Iteration 进行平滑 Linear Warmup
    - 后期：按 Epoch 进行 Multi-Step 阶跃衰减 (默认在 80% 和 90% 处衰减)
    
    Args:
        optimizer: 优化器
        warmup_steps: Warmup 的迭代次数 (通常设为 500 ~ 1000)
        epochs: 总训练 Epoch 数
        steps_per_epoch: 每个 Epoch 包含的 Batch 数量 (即 len(dataloader))
    """
    # 设定阶跃衰减的节点 (转换为具体的 step 数)
    # 如果是短周期微调 (<=12 epoch)，通常只在倒数第一或第二 epoch 降一次 LR
    if epochs <= 12:
        drop_epochs = [epochs - 1]  
    else:
        drop_epochs = [int(epochs * 0.8), int(epochs * 0.9)] 
        
    drop_steps = [e * steps_per_epoch for e in drop_epochs]

    def lr_lambda(current_step: int):
        # ==========================================
        # 1. 阶段一：基于 Iteration 的平滑 Warmup
        # ==========================================
        if current_step < warmup_steps:
            # 防止第 0 步学习率为绝对的 0（避免死掉），给定一个极小的起点如 0.01 倍
            return max(0.01, float(current_step) / float(max(1, warmup_steps)))
        
        # ==========================================
        # 2. 阶段二：基于 Epoch 的 Multi-Step 衰减
        # ==========================================
        gamma = 1.0
        for drop_step in drop_steps:
            if current_step >= drop_step:
                gamma *= 0.1  # 每次路过节点，学习率降为 1/10
                
        return gamma

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


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
    log_interval=10, 
    all_caption_embeddings=None,
    caption_to_idx=None,
):
    """训练一个 epoch，适配动态 loss_dict 的 VL-RT-DETR"""
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    # 文本嵌入逻辑
    if all_caption_embeddings is None and caption_to_idx is None:
        tokenlevel_embdedding = True
    else:
        tokenlevel_embdedding = False
        
    model.train()
    
    # 动态字典用于累计 Loss
    total_loss_sum = 0.0
    accumulated_loss_dict = {}
    
    current_lr = optimizer.param_groups[0]['lr']
        
    if is_main_process:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    global_step = (epoch - 1) * len(dataloader) if hasattr(dataloader, '__len__') else 0
    
    for batch_idx, batch in enumerate(pbar):
        imgs = batch['img'].to(device)
        B = imgs.shape[0]
        batch_captions = batch.get('batch_captions', [])
        batch_captions = [s[:300] for s in batch_captions]
        # 文本特征提取
        if tokenlevel_embdedding:
            textfeats, mask = textencoder.embedtext(batch_captions, normalize=True, batch_size=len(batch_captions), tokenlevel=tokenlevel_embdedding)
            textfeats = textfeats.to(device)
            mask = mask.to(device)
        else:
            indices = [caption_to_idx[cap] for cap in batch_captions]
            textfeats = all_caption_embeddings[indices]
            textfeats = textfeats.view(B, -1, textfeats.shape[-1]).to(device)
            mask = None
        
        batch_data = {
            'batch_idx': batch['batch_idx'].to(device),
            'cls': batch['cls'].to(device),
            'bboxes': batch['bboxes'].to(device),
            'gt_groups': batch['gt_groups'], # 这是 list，不需要 to(device)
            'text_is_positive': batch['text_is_positive'].to(device) if 'text_is_positive' in batch else None,
        }
        
        optimizer.zero_grad()
        
        # ==========================================
        # 前向传播 (匹配 vlrtdetrnet 接口)
        # ==========================================
        with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            preds, loss, loss_dict = model(imgs, textfeats, mask, batch_data)
        
        # ==========================================
        # 反向传播
        # ==========================================
        if use_amp and scaler is not None:
            # 1. 缩放 Loss 并反向传播
            scaler.scale(loss).backward()
            
            # 2. 梯度反缩放 (为梯度裁剪做准备)
            scaler.unscale_(optimizer)
            
            # 3. DETR 必备：梯度裁剪 (防止 Transformer 初期梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            # 4. 记录当前的 Scale 大小
            scale_before = scaler.get_scale()
            
            # 5. 更新优化器 (内部会自动决定是否真实执行 optimizer.step())
            scaler.step(optimizer)
            
            # 6. 更新缩放器 (处理 inf/nan 动态调整 scale 大小)
            scaler.update()
            
            # 如果发生了溢出导致 Scale 缩小，说明 optimizer 被跳过了
            # 这时我们也跳过 scheduler.step()，保持两者步调绝对一致
            scale_after = scaler.get_scale()
            optimizer_was_stepped = (scale_before <= scale_after)
            
            if optimizer_was_stepped:
                scheduler.step()

            # ==========================================
            # 遍历所有要求梯度但没拿到梯度的参数
            # ==========================================
            # if batch_idx == 0: # 只在第一个 batch 打印一次
            #     print("\n🚨 发现以下参数未参与前向传播：")
            #     ghost_count = 0
            #     for name, p in model.named_parameters():
            #         if p.requires_grad and p.grad is None:
            #             print(f"- {name}")
            #             ghost_count += 1
            #     print(f"总计: {ghost_count} 个摸鱼参数\n")
            # DETR 必备：梯度裁剪 (防止 Transformer 初期梯度爆炸)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()

        optimizer.zero_grad()
        
        # ==========================================
        # 累积损失 (解包字典)
        # ==========================================
        loss_val = loss.item()
        total_loss_sum += loss_val
        
        for k, v in loss_dict.items():
            if k not in accumulated_loss_dict:
                accumulated_loss_dict[k] = 0.0
            accumulated_loss_dict[k] += v.item()

        # ==========================================
        # 终端进度条 & Wandb 记录
        # ==========================================
        if is_main_process:
            num_batches = batch_idx + 1
            
            # 提取最核心的三个 loss 用于进度条显示
            avg_cls = accumulated_loss_dict.get('loss_class', 0) / num_batches
            avg_box = accumulated_loss_dict.get('loss_bbox', 0) / num_batches
            avg_giou = accumulated_loss_dict.get('loss_giou', 0) / num_batches
            avg_total = total_loss_sum / num_batches
            
            pbar.set_postfix({
                'Loss': f"{avg_total:.3f}",
                'cls': f"{avg_cls:.3f}",
                'box': f"{avg_box:.3f}",
                'giou': f"{avg_giou:.3f}"
            })
            
            # 记录到 wandb (动态展开字典里的所有 Aux 和 DN Loss)
            if (wandb_run is not None) and (batch_idx % log_interval == 0 or batch_idx == len(dataloader) - 1):
                step = global_step + batch_idx
                log_data = {
                    "batch/step": step,
                    "batch/total_loss": loss_val,
                    "batch/lr": current_lr,
                }
                for k, v in loss_dict.items():
                    log_data[f"batch/{k}"] = v.item()
                
                wandb_run.log(log_data, step=step)
    
    # 更新学习率
    if scheduler is not None:
        scheduler.step()
    
    # ==========================================
    # DDP 场景：动态同步所有损失指标
    # ==========================================
    if world_size > 1:
        # 将字典的值提取出来打平成一个 Tensor 进行 all_reduce
        keys = sorted(accumulated_loss_dict.keys())
        sync_values = [total_loss_sum] + [accumulated_loss_dict[k] for k in keys]
        sync_tensor = torch.tensor(sync_values, device=device)
        
        dist.all_reduce(sync_tensor, op=dist.ReduceOp.AVG)
        
        # 重新赋值回字典
        total_loss_sum = sync_tensor[0].item()
        for i, k in enumerate(keys):
            accumulated_loss_dict[k] = sync_tensor[i + 1].item()
    
    # 计算整个 epoch 的平均损失
    num_batches_total = len(dataloader)
    avg_loss = total_loss_sum / num_batches_total
    avg_loss_dict = {k: v / num_batches_total for k, v in accumulated_loss_dict.items()}
    
    return {
        'loss': avg_loss,
        'loss_items': avg_loss_dict, # 返回完整的字典供上层调用打印
        'lr': current_lr,
    }


def save_checkpoint(model, optimizer, epoch, output_dir, is_best=False, is_main_process=True):
    if not is_main_process:
        return
    
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