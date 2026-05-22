
# train_ddp.py
# 使用 DDP (DistributedDataParallel) 进行多卡训练
# 单机多卡 torchrun --nproc_per_node=7 train_ddp.py
# 多机多卡 torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr="主节点IP" --master_port=12345 train_ddp.py

import os
import pickle
import tempfile
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.amp import GradScaler


from dataset.transforms import make_coco_transforms
from dataset.unified_dataset import UnifiedDetectionDataset   
from dataset.build_dataloader import build_dataloader, load_labels
from dataset.textmodelembedder import CLIPTextEmbedder, Qwen3VLEmbeddingTextEmbedder
from utils.train_utils import train_one_epoch, validate_one_epoch, save_checkpoint, load_model, setup_distributed, cleanup_distributed, is_main_process, get_scheduler
from models.expalignet import expalignet
import wandb




def main():
######################Configurations######################
    num_epochs = 30
    text_embed_dim = 768
    batch_size_per_gpu = 48
    num_workers = 4
    target_size = 640
    output_dir = "outputs-qwen2b-768"
    os.makedirs(output_dir, exist_ok=True)
    tokenlevel_embdedding = True
    num_infonce_batch = 20
    backbone_size = "tiny"
    num_classes = num_infonce_batch
    pretrain_model_path = None#"outputs-qwen2b-768/model_last.pth"
    pretrain_backbone = "/root/autodl-tmp/yoloe/third_party/dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    #pretrain_backbone = "/root/autodl-tmp/VLMs/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
    pretrain_text_encoder = "/project/GLS/HJY/yoloe/ViT-L-14.pt"
    pretrain_text_encoder = "/root/autodl-tmp/Qwen3-VL-Embedding-2B"
    cache_file = os.path.join("/root/autodl-tmp/", "ddp_train_dataset_cache.pkl")#tempfile.gettempdir()
    embdedding_cache_file = "dataset/all_caption_embeddings.pt"
    val_interval = 31  # 每多少个 epoch 验证一次
    use_amp = True
    warmup_epochs = 3          # warmup 轮数
    base_lr = 0.002
    weight_decay = 0.025
    use_wandb = True
    wandb_run = None

    # 数据配置
    train_ann_files = [
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
    ]
    train_image_roots = [
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
    ]
    allow_complex_augmentation = [True, True, False, True, False, False, False, False, False, True, False]
    val_ann_files = ["/root/autodl-tmp/OOD/coco/annotations/instances_val2017.json"]
    val_image_roots = ["/root/autodl-tmp/OOD/coco/images/val2017"]

    
######################Building Data Set######################
    # DDP: 主进程构建 dataset 并保存 samples + 嵌入缓存，其他进程从缓存加载
    cache_data = None
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
    # 文本编码器 - 只有主进程需要
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    textencoder = None
    if is_main_process(rank):
        print(f"DDP: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"Batch per GPU: {batch_size_per_gpu}, Total: {batch_size_per_gpu * world_size}")
        print("Building text encoder...")
        #textencoder = CLIPTextEmbedder(pretrain_text_encoder, device=device, mrl_truncate=text_embed_dim)
    textencoder = Qwen3VLEmbeddingTextEmbedder(pretrain_text_encoder, device=device, mrl_truncate=text_embed_dim)
    all_caption_embeddings = None
    caption_to_idx = None

    # 构建主进程训练和验证 DataLoader（仅主进程需要）
    val_loader = None
    if is_main_process(rank):
        if use_wandb:
            wandb_run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="inlmouse-tsinghua-university",
                # Set the wandb project where this run will be logged.
                project="VLMs",
                name="mach",
                # Track hyperparameters and run metadata.
                config={
                    "base_lr": base_lr,
                    "size": backbone_size,
                    "text_embed_dim": text_embed_dim,
                    "target_size": target_size,
                    "num_infonce_batch": num_infonce_batch,
                    "dataset": train_ann_files,
                    "epochs": num_epochs,
                    "warmup_epochs": warmup_epochs,
                    "weight_decay": weight_decay,
                },
            )
        print("[Main] Building training dataset and computing embeddings...")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            # 构建 dataset，samples 从缓存加载，嵌入从文件加载
            train_dataset = UnifiedDetectionDataset(
                samples=cache_data['samples'],
                label_list=cache_data['label_list'],
                num_infonce_batch=num_infonce_batch,
                transforms=make_coco_transforms(istrain=True, target_size=target_size),
                istrain=True,
            )
        else:
            temp_loader = build_dataloader(
                ann_files=train_ann_files,
                image_roots=train_image_roots,
                allow_complex_augmentation = allow_complex_augmentation,
                num_infonce_batch=num_infonce_batch,
                batch_size=batch_size_per_gpu,
                shuffle=True,
                num_workers=num_workers,
                istrain=True,
                target_size=target_size,
                labels_file=None,
            )
            
            # 构建完整 dataset，会计算并保存嵌入到本地文件
            train_dataset=temp_loader.dataset
            
            # 保存 samples 到共享缓存文件，供非主进程加载
            cache_data = {
                'samples': train_dataset.samples,
                'label_list': train_dataset.label_list,
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            if not tokenlevel_embdedding:
                if os.path.exists(embdedding_cache_file):
                    print(f"[Main] 从缓存{embdedding_cache_file}加载 caption 嵌入")
                    embdedding_cache_data = torch.load(embdedding_cache_file, map_location='cpu')
                    all_caption_embeddings, caption_to_idx = embdedding_cache_data
                    all_captions_list = list(caption_to_idx.keys())
                else:
                    # 从 samples 收集所有 caption
                    all_captions_set = set()
                    for sample in train_dataset.samples:
                        all_captions_set.update(sample.get("captions", []))
                    all_captions_list = sorted(list(all_captions_set))
                    print(f"[Main] 预计算 {len(all_captions_list)} 个 grounding caption 嵌入...")
                    all_caption_embeddings, _ = textencoder.embedtext(all_captions_list, normalize=True, batch_size=64, tokenlevel=tokenlevel_embdedding)
                    caption_to_idx = {cap: i for i, cap in enumerate(all_captions_list)}
                    torch.save((all_caption_embeddings.cpu(), caption_to_idx), embdedding_cache_file)
                    print(f"[Main] Caption embeddings 已缓存到 {embdedding_cache_file}")
            print(f"[Rank {rank}] Main: dataset ready, cache saved to {cache_file}")
            del temp_loader  # 释放临时 loader 占用的资源

        print("[Main] Building validation dataloader...")
        val_loader = build_dataloader(
            ann_files=val_ann_files,
            image_roots=val_image_roots,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            istrain=False,
            target_size=target_size,
            labels_file=f"{output_dir}/val_labels.txt",
        )
        label_list = load_labels(f"{output_dir}/val_labels.txt")
        txt_feats, _ = textencoder.embedtext(label_list, normalize=True, batch_size=32, tokenlevel=tokenlevel_embdedding)


    # Barrier: 等待主进程完成缓存保存
    if world_size > 1:
        dist.barrier()
    
    # 非主进程：从缓存文件加载 samples 和嵌入
    if not is_main_process(rank):
        print(f"[Rank {rank}] Worker: loading dataset from cache...")
        
        if cache_data is None:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
        if not tokenlevel_embdedding:
            if os.path.exists(embdedding_cache_file):
                print(f"[Rank {rank}] Worker: 从缓存{embdedding_cache_file}加载 caption 嵌入")
                embdedding_cache_data = torch.load(embdedding_cache_file, map_location='cpu')
                all_caption_embeddings, caption_to_idx = embdedding_cache_data
                all_captions_list = list(caption_to_idx.keys())
            else:
                assert False

        # 构建 dataset，samples 从缓存加载，嵌入从文件加载
        train_dataset = UnifiedDetectionDataset(
            samples=cache_data['samples'],
            label_list=cache_data['label_list'],
            num_infonce_batch=num_infonce_batch,
            transforms=make_coco_transforms(istrain=True, target_size=target_size),
            istrain=True,
        )
        print(f"[Rank {rank}] Worker: dataset ready")
    
    # 主进程清理缓存文件
    if world_size > 1:
        dist.barrier()
    # if is_main_process(rank) and os.path.exists(cache_file):
    #     os.remove(cache_file)
    
    # 再次 barrier 确保所有进程同步
    if world_size > 1:
        dist.barrier()
    
    # 构建带 DistributedSampler 的 DataLoader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,  # sampler handles shuffle
        num_workers=num_workers,
        istrain=True,
        target_size=target_size,
        sampler=train_sampler,
        pin_memory=True,
    )
    
######################## Building Model and Optimizer ##################
    if is_main_process(rank):
        #del textencoder  # Release textencoder memory in main process after dataset is built and cached
        print("Building model...")

    if pretrain_model_path is not None:
        model, optimizer, start_epoch = load_model(pretrain_model_path, text_embed_dim, num_classes, device)
    else:
        model = expalignet(
            text_embed_dim = text_embed_dim,
            size=backbone_size,
            pretrained=pretrain_backbone,
            num_classes=num_classes
        )
        model.to(device)
        #optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': base_lr},
            {'params': model.neck.parameters(), 'lr': base_lr},
            {'params': [p for n, p in model.head.named_parameters() if 'alignhead' not in n.lower()], 'lr': base_lr},
            {'params': model.head.alignhead.parameters(), 'lr': base_lr * 1},     # BNContrastiveHead
        ], weight_decay=weight_decay)
        start_epoch = 1

    scheduler = get_scheduler(optimizer, warmup_epochs, num_epochs, start_epoch)
      
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    scaler = GradScaler() if use_amp else None

    if is_main_process(rank):
        print(f"Training samples: {len(train_dataset)}")
        if start_epoch > 1:
            print(f"Resuming from epoch {start_epoch} with checkpoint {pretrain_model_path}")
        else:
            print("Starting training from scratch...")
    
    
    
######################Start Training Loop######################
    best_map = 0.0
    is_best = False
    # 训练循环
    for epoch in range(start_epoch - 1, num_epochs):
        # -------- Training --------
        train_results = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            device=device,
            scheduler=scheduler,  
            train_sampler=train_sampler,
            use_amp=use_amp,
            is_main_process=is_main_process(rank),
            textencoder=textencoder,
            world_size=world_size,
            wandb_run=wandb_run,
        )
        
        # -------- Validation（使用统一的验证函数） --------
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            val_results = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                epoch=epoch + 1,
                txt_feats=txt_feats,
                label_list=label_list,
                target_size=target_size,
                compute_loss=False,  # DDP 场景下不计算损失，减少显存
                is_main_process=is_main_process(rank),
                textencoder=textencoder if is_main_process(rank) else None,  # 只有主进程需要文本编码器
                output_dir=output_dir 
            )
            
            # Finding best model based on mAP (only main process does this)
            if is_main_process(rank):
                map_50_95 = val_results['map']
                if map_50_95 > best_map:
                    is_best = True
                    best_map = map_50_95
                else:
                    is_best = False
        
        # -------- Save last checkpoint each epoch --------
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            output_dir=output_dir,
            is_best=is_best,
            is_main_process=is_main_process(rank),
        )
    if use_wandb and is_main_process(rank):
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
