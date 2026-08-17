
# train_ddp.py
# Multi-GPU training with DDP (DistributedDataParallel)
# Single-node:  torchrun --nproc_per_node=8 train_ddp.py
# Multi-node:   torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr="<master-ip>" --master_port=12345 train_ddp.py

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
# 1. Centralized configuration (Dataclass)
# ==========================================
@dataclass
class TrainConfig:
    # Basic settings
    model_name = "expalignet"
    num_epochs: int = 30
    batch_size_per_gpu: int = 16
    num_workers: int = 4
    target_size: int = 640
    output_dir: str = "outputs-qwen2b-768"
    
    # Model and feature settings
    text_embed_dim: int = 768
    tokenlevel_embdedding: bool = True
    num_infonce_batch: int = 20
    backbone_size: str = "tiny"
    num_classes: int = num_infonce_batch  # equal to num_infonce_batch
    
    # Pretrained weight paths
    pretrain_model_path: Optional[str] = None#"outputs-qwen2b-768/model_tcem_epoch1.pth"
    resume: bool = False  # whether to resume training
    pretrain_backbone: str = "/path/to/dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    pretrain_text_encoder: str = "/path/to/Qwen3-VL-Embedding-2B"
    
    # Cache settings
    cache_file: str = "/path/to/ddp_train_dataset_cache.pkl"
    embdedding_cache_file: str = "dataset/all_caption_embeddings.pt"
    
    # Optimizer and training strategy
    val_interval: int = 1
    use_amp: bool = True
    warmup_epochs: int = 0
    base_lr: float = 2e-3#1e-4#
    weight_decay: float = 0.025#1e-4#
    use_wandb: bool = True
    wandb_project: str = "VLMs"
    wandb_entity: str = "N/A"
    wandb_run_name: str = "mach+jepa"

    # Dataset settings
    train_ann_files: List[str] = field(default_factory=lambda: [
        "/path/to/Objects365_v1/annotations/objects365_train_with_caption.json",
        "/path/to/flickr30k/final_flickr_separateGT_train_segm.json",
        "/path/to/MixedGrounding/mdetr_annotations/final_mixed_train_no_coco_segm_fixed.json",
    ])
    train_image_roots: List[str] = field(default_factory=lambda: [
        "/path/to/Objects365_v1/images/train",
        "/path/to/flickr30k/flickr30k-images",
        "/path/to/MixedGrounding/images",
    ])
    allow_complex_augmentation: List[bool] = field(default_factory=lambda: [
        False, False, False
    ])

    val_ann_files: List[str] = field(default_factory=lambda: ["/path/to/coco/annotations/instances_val2017.json"])
    val_image_roots: List[str] = field(default_factory=lambda: ["/path/to/coco/images/val2017"])

# ==========================================
# 2. Core modular functions
# ==========================================

def setup_wandb(config: TrainConfig):
    """Initialize Wandb"""
    return wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.__dict__
    )

def prepare_dataset(config: TrainConfig, rank: int, world_size: int, text_encoder):
    """Prepare dataset and caches: built by the main process, loaded by the other ranks"""
    cache_data = None
    all_caption_embeddings = None
    caption_to_idx = None
    all_captions_list = None

    # ==================================
    # [Main process] Build data and caches
    # ==================================
    if is_main_process(rank):
        print("[Main] Building training dataset and computing embeddings...")
        
        # 1. Dataset cache
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

        # 2. Caption embedding cache (read or generated by the main process)
        if not config.tokenlevel_embdedding:
            if os.path.exists(config.embdedding_cache_file):
                print(f"[Main] Loading caption embeddings from cache {config.embdedding_cache_file}")
                embdedding_cache_data = torch.load(config.embdedding_cache_file, map_location='cpu')
                all_caption_embeddings, caption_to_idx = embdedding_cache_data
                all_captions_list = list(caption_to_idx.keys())
            else:
                all_captions_set = {cap for sample in cache_data['samples'] for cap in sample.get("captions", [])}
                all_captions_list = sorted(list(all_captions_set))
                print(f"[Main] Precomputing {len(all_captions_list)} caption embeddings...")
                
                all_caption_embeddings, _ = text_encoder.embedtext(
                    all_captions_list, normalize=True, batch_size=64, tokenlevel=False
                )
                caption_to_idx = {cap: i for i, cap in enumerate(all_captions_list)}
                
                torch.save((all_caption_embeddings.cpu(), caption_to_idx), config.embdedding_cache_file)
                print(f"[Main] Caption embeddings cached to {config.embdedding_cache_file}")
                

    # Barrier: wait for the main process to finish building and writing the caches
    if world_size > 1:
        dist.barrier()

    # ==================================
    # [Worker ranks] Load from cache
    # ==================================
    if not is_main_process(rank):
        print(f"[Rank {rank}] Loading dataset from cache...")
        
        # 1. Load dataset cache
        with open(config.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            
        # 2. Load caption embedding cache
        if not config.tokenlevel_embdedding:
            if os.path.exists(config.embdedding_cache_file):
                print(f"[Rank {rank}] Worker: loading caption embeddings from cache {config.embdedding_cache_file}")
                embdedding_cache_data = torch.load(config.embdedding_cache_file, map_location='cpu')
                all_caption_embeddings, caption_to_idx = embdedding_cache_data
                all_captions_list = list(caption_to_idx.keys())
            else:
                assert False, f"Embedding cache not found for worker rank {rank}!"

    # ==================================
    # Build Dataset and DataLoader
    # ==================================
    train_dataset = UnifiedDetectionDataset(
        samples=cache_data['samples'],
        label_list=cache_data['label_list'],
        num_infonce_batch=config.num_infonce_batch,
        transforms=make_coco_transforms(istrain=True, target_size=config.target_size),
        istrain=True,
    )
    
    # Note: if UnifiedDetectionDataset needs the embeddings, pass them in here
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


def build_training_model(config, device: torch.device, local_rank: int, world_size: int, steps_per_epoch: int):
    """Build model, optimizer and scheduler (compatible with the factory and multiple architectures)"""
    
    # 1. Extract the config (dataclass or argparse) into a standard args dict for the factory
    model_args = {
        'model_name': getattr(config, 'model_name', 'vlrtdetrnet'), # default to the new architecture
        'text_embed_dim': config.text_embed_dim,
        'size': getattr(config, 'backbone_size', 'tiny'),
        'num_classes': config.num_classes,
        'pretrained': getattr(config, 'pretrain_backbone', None),
        'reg_max': getattr(config, 'reg_max', 16) # backward compatibility with expalignet
    }

    # Resume switch (if not set in config, a pretrained checkpoint is treated as fine-tuning)
    resume_mode = getattr(config, 'resume', False)

    # ==========================================
    # 2. Core loading logic (from-scratch / resume / fine-tune and optimizer param groups in one call)
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
    # 3. Learning rate scheduler
    # ==========================================
    scheduler = get_scheduler(
        optimizer=optimizer, 
        warmup_steps=min(1000, steps_per_epoch), 
        epochs=config.num_epochs, 
        steps_per_epoch=steps_per_epoch
    )
    
    # ==========================================
    # 4. DDP wrapping and throughput optimization
    # ==========================================
    if world_size > 1:
        # Tip: vlrtdetrnet has no parameters that stay unused in the forward pass, so
        # disabling find_unused_parameters saves DDP from traversing the graph every
        # backward step and improves throughput. The legacy expalignet may still need it.
        need_find_unused = (model_args['model_name'] == 'expalignet')
        
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank, 
            find_unused_parameters=need_find_unused
        )
        
    return model, optimizer, scheduler, start_epoch


# ==========================================
# 3. Main loop
# ==========================================
def main():
    config = TrainConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Environment and DDP initialization
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process(rank):
        print(f"DDP Config: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"Batch Size: {config.batch_size_per_gpu} per GPU, Total: {config.batch_size_per_gpu * world_size}")

    wandb_run = setup_wandb(config) if (config.use_wandb and is_main_process(rank)) else None

    # 2. Text encoder initialization
    text_encoder = None
    #if is_main_process(rank):
    text_encoder = Qwen3VLEmbeddingTextEmbedder(config.pretrain_text_encoder, device=device, mrl_truncate=config.text_embed_dim)

    # 3. Prepare data
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

    
    model, optimizer, scheduler, start_epoch = build_training_model(config, device, local_rank, world_size, steps_per_epoch=len(train_loader))

    scaler = GradScaler() if config.use_amp else None

    # 5. Training loop
    best_map = 0.0
    for epoch in range(start_epoch - 1, config.num_epochs):
        current_epoch = epoch + 1
        
        # --- Training phase ---
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
        
        # --- Validation phase ---
        is_best = False
        # --- Save checkpoint ---
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
                # RefCOCOg-specific
                coco_json_path="/path/to/refcoco/annotations/refcoco_testA_reviewed.json",
                image_root="/path/to/refcoco/images/train2014",
                conf_thresh=0.01,
                nms_thresh=0.75,
                wandb_run=wandb_run,
            )
            if is_main_process(rank):
                map_50_95 = val_results.get('refcoco_acc', 0.0)
                if map_50_95 > best_map:
                    best_map = map_50_95
                    is_best = True
        
        # break
    # 6. Cleanup and exit
    if wandb_run and is_main_process(rank):
        wandb_run.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()
