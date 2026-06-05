
# train_ddp.py
# 使用 DDP (DistributedDataParallel) 进行多卡训练
# 单机多卡 torchrun --nproc_per_node=8 train_ddp.py
# 多机多卡 torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr="主节点IP" --master_port=12345 train_ddp.py

import os
import pickle
import tempfile
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import List, Optional

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.amp import GradScaler


from dataset.transforms import make_coco_transforms
from dataset.unified_dataset import UnifiedDetectionDataset   
from dataset.build_dataloader import build_dataloader, load_labels
from dataset.textmodelembedder import CLIPTextEmbedder, Qwen3VLEmbeddingTextEmbedder
from utils.detr_train_utils import train_one_epoch, save_checkpoint, load_model, setup_distributed, cleanup_distributed, is_main_process, get_scheduler
from evaluation.refcoco_reviewed_online_evaluator import validate_refcoco_one_epoch
import wandb


# ==========================================
# 1. 集中管理配置项 (Dataclass)
# ==========================================
@dataclass
class TrainConfig:
    # 基础配置
    model_name = "vlrtdetrnet"
    num_epochs: int = 30
    batch_size_per_gpu: int = 48
    num_workers: int = 4
    target_size: int = 640
    output_dir: str = "outputs-qwen2b-768"
    
    # 模型与特征配置
    text_embed_dim: int = 768
    tokenlevel_embdedding: bool = True
    num_infonce_batch: int = 20
    backbone_size: str = "tiny"
    num_classes: int = num_infonce_batch  # 等于 num_infonce_batch
    
    # 预训练权重路径
    pretrain_model_path: Optional[str] = None#"outputs-qwen2b-768/model_last.pth"
    pretrain_backbone: str = "/root/autodl-tmp/yoloe/third_party/dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    pretrain_text_encoder: str = "/root/autodl-tmp/Qwen3-VL-Embedding-2B"
    
    # 缓存配置
    cache_file: str = "/root/autodl-tmp/ddp_train_dataset_cache.pkl"
    embdedding_cache_file: str = "dataset/all_caption_embeddings.pt"
    
    # 优化器与训练策略
    val_interval: int = 1
    use_amp: bool = True
    warmup_epochs: int = 3
    base_lr: float = 0.002
    weight_decay: float = 0.025
    use_wandb: bool = False
    wandb_project: str = "VLMs"
    wandb_entity: str = "inlmouse-tsinghua-university"
    wandb_run_name: str = "vlrtdetrnet"

    # 数据集配置
    train_ann_files: List[str] = field(default_factory=lambda: [
        # "/root/autodl-tmp/OOD/Objects365_v2/annotations/zhiyuan_objv2_train_fixname_with_caption.json",
        "/root/autodl-tmp/OOD/DeepFashion2/annotations/deepfasion2_with_caption.json",
        "/root/autodl-tmp/OOD/coco/annotations/lvis_v1_train.json",
        "/root/autodl-tmp/OOD/Objects365_v1/annotations/objects365_train_with_caption.json",
        "/root/autodl-tmp/OOD/Objects365_v1/annotations/objects365_train_segm.json",
        "/root/autodl-tmp/OOD/flickr30k/final_flickr_separateGT_train_segm.json",
        "/root/autodl-tmp/OOD/MixedGrounding/mdetr_annotations/final_mixed_train_no_coco_segm_fixed.json",
        "/root/autodl-tmp/OOD/refcoco/annotations/grefcoco_sgem.json",
        "/root/autodl-tmp/OOD/refcoco/annotations/reircoco_train_segm.json",
        "/root/autodl-tmp/OOD/refcoco/annotations/refcoco_unc_train_coco_segm.json",
        "/root/autodl-tmp/OOD/refcoco/annotations/refcoco+_unc_train_coco_segm.json",
        "/root/autodl-tmp/OOD/refcoco/annotations/refcocog_umd_train_coco_segm.json",
    ])
    train_image_roots: List[str] = field(default_factory=lambda: [
        # "/root/autodl-tmp/OOD/Objects365_v2/images/train",
        "/root/autodl-tmp/OOD/DeepFashion2/images",
        "/root/autodl-tmp/OOD/coco/images/train2017",
        "/root/autodl-tmp/OOD/Objects365_v1/images/train",
        "/root/autodl-tmp/OOD/Objects365_v1/images/train",
        "/root/autodl-tmp/OOD/flickr30k/flickr30k-images",
        "/root/autodl-tmp/OOD/MixedGrounding/images",
        "/root/autodl-tmp/OOD/refcoco/images/train2014",
        "/root/autodl-tmp/OOD/refcoco/images/train2014",
        "/root/autodl-tmp/OOD/refcoco/images/train2014",
        "/root/autodl-tmp/OOD/refcoco/images/train2014",
        "/root/autodl-tmp/OOD/refcoco/images/train2014",
    ])
    allow_complex_augmentation: List[bool] = field(default_factory=lambda: [
        True, True, False, True, False, False, False, False, False, True, False
    ])
    # allow_complex_augmentation: List[bool] = field(default_factory=lambda: [ False ])
    val_ann_files: List[str] = field(default_factory=lambda: ["/root/autodl-tmp/OOD/coco/annotations/instances_val2017.json"])
    val_image_roots: List[str] = field(default_factory=lambda: ["/root/autodl-tmp/OOD/coco/images/val2017"])

# ==========================================
# 2. 核心模块化函数
# ==========================================

def setup_wandb(config: TrainConfig):
    """初始化 Wandb"""
    return wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.__dict__
    )

def prepare_dataset(config: TrainConfig, rank: int, world_size: int, text_encoder):
    """准备数据集和缓存：主进程生成，子进程加载"""
    cache_data = None
    all_caption_embeddings = None
    caption_to_idx = None
    all_captions_list = None

    # ==================================
    # [主进程] 处理数据并建立缓存
    # ==================================
    if is_main_process(rank):
        print("[Main] Building training dataset and computing embeddings...")
        
        # 1. Dataset 缓存
        if os.path.exists(config.cache_file):
            with open(config.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
        else:
            temp_loader = build_dataloader(
                ann_files=config.train_ann_files,
                image_roots=config.train_image_roots,
                allow_complex_augmentation=config.allow_complex_augmentation,
                num_infonce_batch=config.num_infonce_batch,
                batch_size=config.batch_size_per_gpu,
                shuffle=True,
                num_workers=config.num_workers,
                istrain=True,
                target_size=config.target_size,
                labels_file=None,
            )
            cache_data = {
                'samples': temp_loader.dataset.samples,
                'label_list': temp_loader.dataset.label_list,
            }
            with open(config.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            del temp_loader
        print(f"[Main] Dataset ready, cached labels to {config.cache_file}")

        # 2. Caption Embedding 缓存 (主进程读取或生成)
        if not config.tokenlevel_embdedding:
            if os.path.exists(config.embdedding_cache_file):
                print(f"[Main] 从缓存 {config.embdedding_cache_file} 加载 caption 嵌入")
                embdedding_cache_data = torch.load(config.embdedding_cache_file, map_location='cpu')
                all_caption_embeddings, caption_to_idx = embdedding_cache_data
                all_captions_list = list(caption_to_idx.keys())
            else:
                all_captions_set = {cap for sample in cache_data['samples'] for cap in sample.get("captions", [])}
                all_captions_list = sorted(list(all_captions_set))
                print(f"[Main] 预计算 {len(all_captions_list)} 个 caption embeddings...")
                
                all_caption_embeddings, _ = text_encoder.embedtext(
                    all_captions_list, normalize=True, batch_size=64, tokenlevel=False
                )
                caption_to_idx = {cap: i for i, cap in enumerate(all_captions_list)}
                
                torch.save((all_caption_embeddings.cpu(), caption_to_idx), config.embdedding_cache_file)
                print(f"[Main] Caption embeddings 已缓存到 {config.embdedding_cache_file}")
                

    # Barrier：等待主进程完成数据集和嵌入的计算与缓存写入
    if world_size > 1:
        dist.barrier()

    # ==================================
    # [子进程] 读取缓存
    # ==================================
    if not is_main_process(rank):
        print(f"[Rank {rank}] Loading dataset from cache...")
        
        # 1. Dataset 缓存读取
        with open(config.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            
        # 2. Caption Embedding 缓存读取
        if not config.tokenlevel_embdedding:
            if os.path.exists(config.embdedding_cache_file):
                print(f"[Rank {rank}] Worker: 从缓存 {config.embdedding_cache_file} 加载 caption 嵌入")
                embdedding_cache_data = torch.load(config.embdedding_cache_file, map_location='cpu')
                all_caption_embeddings, caption_to_idx = embdedding_cache_data
                all_captions_list = list(caption_to_idx.keys())
            else:
                assert False, f"Embedding cache not found for worker rank {rank}!"

    # ==================================
    # 构建 Dataset 和 DataLoader
    # ==================================
    train_dataset = UnifiedDetectionDataset(
        samples=cache_data['samples'],
        label_list=cache_data['label_list'],
        num_infonce_batch=config.num_infonce_batch,
        transforms=make_coco_transforms(istrain=True, target_size=config.target_size),
        istrain=True,
    )
    
    # 注意：如果 UnifiedDetectionDataset 需要接收 embeddings，记得在这里传入
    # train_dataset.all_caption_embeddings = all_caption_embeddings 
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size_per_gpu,
        shuffle=False,  
        num_workers=config.num_workers,
        istrain=True,
        target_size=config.target_size,
        sampler=train_sampler,
        pin_memory=True,
    )
    
    return train_loader, train_sampler, all_caption_embeddings, caption_to_idx


def build_training_model(config, device: torch.device, local_rank: int, world_size: int):
    """构建模型、优化器和调度器 (完美适配 Factory 与多架构)"""
    
    # 1. 把 config (可能是 dataclass 或 argparse) 提取为标准的 args 字典供 Factory 使用
    model_args = {
        'model_name': getattr(config, 'model_name', 'vlrtdetrnet'), # 默认使用新架构
        'text_embed_dim': config.text_embed_dim,
        'size': getattr(config, 'backbone_size', 'tiny'),
        'num_classes': config.num_classes,
        'pretrained': getattr(config, 'pretrain_backbone', None),
        'reg_max': getattr(config, 'reg_max', 16) # 兼容旧版 expalignet
    }

    # 获取续训开关 (假设你在 config 里配了这个参数，没配则默认如果是预训练权重就做微调)
    resume_mode = getattr(config, 'resume', False)

    # ==========================================
    # 2. 核心加载逻辑 (一行代码搞定从头训/续训/微调、优化器分组)
    # ==========================================
    model, optimizer, start_epoch = load_model(
        ckpt_path=config.pretrain_model_path,
        args=model_args,
        device=device,
        base_lr=config.base_lr,
        weight_decay=config.weight_decay,
        resume=resume_mode
    )

    # ==========================================
    # 3. 学习率调度器
    # ==========================================
    scheduler = get_scheduler(
        optimizer=optimizer, 
        warmup_epochs=config.warmup_epochs, 
        num_epochs=config.num_epochs, 
        start_epoch=start_epoch
    )
    
    # ==========================================
    # 4. DDP 多卡包装与极致加速
    # ==========================================
    if world_size > 1:
        # 💡 核心优化：vlrtdetrnet 已经被我们清理干净，没有任何未参与前向传播的游离参数。
        # 关闭 find_unused_parameters 可以省去 DDP 每次反向传播时遍历计算图寻找废弃参数的时间，极大提升吞吐量！
        # 但如果是旧版的 expalignet，可能还需要开着。
        need_find_unused = (model_args['model_name'] == 'expalignet')
        
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank, 
            find_unused_parameters=need_find_unused
        )
        
    return model, optimizer, scheduler, start_epoch


# ==========================================
# 3. 主循环
# ==========================================
def main():
    config = TrainConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. 环境与 DDP 初始化
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process(rank):
        print(f"DDP Config: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"Batch Size: {config.batch_size_per_gpu} per GPU, Total: {config.batch_size_per_gpu * world_size}")

    wandb_run = setup_wandb(config) if (config.use_wandb and is_main_process(rank)) else None

    # 2. 文本编码器初始化
    text_encoder = None
    #if is_main_process(rank):
    text_encoder = Qwen3VLEmbeddingTextEmbedder(config.pretrain_text_encoder, device=device, mrl_truncate=config.text_embed_dim)

    # 3. 准备数据
    all_caption_embeddings = None
    caption_to_idx = None
    train_loader, train_sampler, all_caption_embeddings, caption_to_idx = prepare_dataset(config, rank, world_size, text_encoder)
    
    val_loader, txt_feats, label_list = None, None, None
    if is_main_process(rank):
        print("[Main] Building validation dataloader...")
        val_labels_path = os.path.join(config.output_dir, "val_labels.txt")
        val_loader = build_dataloader(
            ann_files=config.val_ann_files,
            image_roots=config.val_image_roots,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            istrain=False,
            target_size=config.target_size,
            labels_file=val_labels_path,
        )
        label_list = load_labels(val_labels_path)
        txt_feats, _ = text_encoder.embedtext(label_list, normalize=True, batch_size=32, tokenlevel=config.tokenlevel_embdedding)

    
    model, optimizer, scheduler, start_epoch = build_training_model(config, device, local_rank, world_size)

    scaler = GradScaler() if config.use_amp else None

    # 5. 训练主循环
    best_map = 0.0
    for epoch in range(start_epoch - 1, config.num_epochs):
        current_epoch = epoch + 1
        
        # --- 训练阶段 ---
        train_results = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=current_epoch,
            device=device,
            scheduler=scheduler,  
            train_sampler=train_sampler,
            use_amp=config.use_amp,
            is_main_process=is_main_process(rank),
            textencoder=text_encoder,
            world_size=world_size,
            wandb_run=wandb_run,
            all_caption_embeddings=all_caption_embeddings,
            caption_to_idx=caption_to_idx,
        )
        
        # --- 验证阶段 ---
        is_best = False
        # --- 保存权重 ---
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=current_epoch,
            output_dir=config.output_dir,
            is_best=is_best,
            is_main_process=is_main_process(rank),
        )
        
        if val_loader and (current_epoch % config.val_interval == 0):
            # val_results = validate_one_epoch(
            #     model=model,
            #     dataloader=val_loader,
            #     device=device,
            #     epoch=current_epoch,
            #     txt_feats=txt_feats,
            #     label_list=label_list,
            #     target_size=config.target_size,
            #     compute_loss=False, 
            #     is_main_process=is_main_process(rank),
            #     textencoder=text_encoder if is_main_process(rank) else None,
            #     output_dir=config.output_dir 
            # )
            val_results = validate_refcoco_one_epoch(
                model=model,
                device=device,
                epoch=current_epoch,
                target_size=config.target_size,
                is_main_process=is_main_process(rank),
                textencoder=text_encoder if is_main_process(rank) else None,
                output_dir=config.output_dir,
                # RefCOCOg 专用
                coco_json_path="/root/autodl-tmp/OOD/refcoco/annotations/refcocog_val_reviewed.json",
                image_root="/root/autodl-tmp/OOD/refcoco/images/train2014",
                conf_thresh=0.01,
                nms_thresh=0.75,
            )
            if is_main_process(rank):
                map_50_95 = val_results.get('map', 0.0)
                if map_50_95 > best_map:
                    best_map = map_50_95
                    is_best = True
        
        # break
    # 6. 清理退出
    if wandb_run and is_main_process(rank):
        wandb_run.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()