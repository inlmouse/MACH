import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import os
from models.expalignet import expalignet
from dataset.build_dataloader import build_dataloader, load_labels
from dataset.textmodelembedder import Qwen3VLEmbeddingTextEmbedder, CLIPTextEmbedder
from utils.cnn_train_utils import train_one_epoch, validate_one_epoch, save_checkpoint, load_model, get_scheduler



# 主训练函数
def train():
######################Configurations######################
    num_epochs = 25
    text_embed_dim = 1024
    batch_size_per_gpu = 16
    num_workers = 0
    target_size = 640
    output_dir = "outputs-qwen2b-1024"
    os.makedirs(output_dir, exist_ok=True)
    num_infonce_batch = 80
    backbone_size = "base"
    num_classes = num_infonce_batch
    pretrain_model_path = None#"outputs-qwen2b-768/model_last.pth"
    pretrain_backbone = "/root/autodl-tmp/VLMs/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
    pretrain_text_encoder = "/root/autodl-tmp/Qwen3-VL-Embedding-2B"
    #pretrain_text_encoder = "/project/GLS/HJY/yoloe/ViT-L-14.pt"
    val_interval = 1  # 每 5 个 epoch 验证一次
    use_amp = True
    warmup_epochs = 3          # warmup 轮数
    base_lr = 0.002
    weight_decay = 0.025

    # 数据配置
    train_ann_files = [
        #"/data/OOD/coco/annotations/instances_val2017.json",
        "/root/autodl-tmp/OOD/coco/annotations/lvis_v1_train.json",
        "/root/autodl-tmp/OOD/Objects365_v1/annotations/objects365_train_with_caption.json",
        "/root/autodl-tmp/OOD/Objects365_v1/annotations/objects365_train_segm.json",
        "/root/autodl-tmp/OOD/flickr30k/final_flickr_separateGT_train_segm.json",
        "/root/autodl-tmp/OOD/MixedGrounding/mdetr_annotations/final_mixed_train_no_coco_segm.json",
        "/root/autodl-tmp/OOD/refcoco/annotations/grefcoco_sgem.json",
        "/root/autodl-tmp/OOD/refcoco/annotations/reircoco_train_segm.json",
        "/root/autodl-tmp/OOD/refcoco/annotations/refcoco_unc_train_coco_segm.json",
        "/root/autodl-tmp/OOD/refcoco/annotations/refcoco+_unc_train_coco_segm.json",
        "/root/autodl-tmp/OOD/refcoco/annotations/refcocog_umd_train_coco_segm.json",
    ]
    train_image_roots = [
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
    
    val_ann_files = ["/root/autodl-tmp/OOD/coco/annotations/instances_val2017.json"]
    val_image_roots = ["/root/autodl-tmp/OOD/coco/images/val2017"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################Building Data Set######################
    textencoder = Qwen3VLEmbeddingTextEmbedder(pretrain_text_encoder, device=device, mrl_truncate=text_embed_dim)

    # -------- 构建 train_loader --------
    train_loader = build_dataloader(
        ann_files=train_ann_files,
        image_roots=train_image_roots,
        num_infonce_batch=num_infonce_batch,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=num_workers,
        istrain=True,
        target_size=target_size,
        textembedder=textencoder,  # 传入文本嵌入器
    )
    
    # -------- 构建 val_loader --------
    # 验证集使用相同的类别数，istrain=False
    # 验证集也需要 label_list 用于 set_class，从训练集获取
    val_loader = build_dataloader(
        ann_files=val_ann_files,
        image_roots=val_image_roots,
        batch_size=batch_size_per_gpu,
        shuffle=False,  # 验证不需要打乱
        num_workers=num_workers,
        istrain=False,  # 验证模式
        target_size=target_size,
        labels_file=f"{output_dir}/val_labels.txt",
    )
    label_list = load_labels(f"{output_dir}/val_labels.txt")
    txt_feats = textencoder.embedtext(label_list, normalize=True).to(device)

    # 实例化模型
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
        
        optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        start_epoch = 1
    ###
    # model = torch.compile(
    #     model,
    #     mode="max-autotune",   # 推荐
    #     fullgraph=False        # 默认即可
    # )
    ###
    scheduler = get_scheduler(optimizer, warmup_epochs, num_epochs, start_epoch)
    scaler = GradScaler()  # 混合精度训练
    model.head.alignhead.parameters()
######################Start Training Loop######################
    best_map = 0.0
    is_best = False

    for epoch in range(start_epoch - 1, num_epochs):
        # 训练（使用统一的训练函数）
        train_results = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            device=device,
            scheduler=scheduler, 
            train_sampler=None,  # 单卡训练不需要 sampler
            use_amp=use_amp,
            is_main_process=True,
            world_size=1,
            log_interval=10,  # 每 10 个 batch 记录一次
        )

        
        # 验证
        if epoch % val_interval == 0:
            val_results = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                epoch=epoch + 1,
                txt_feats=txt_feats,
                label_list=label_list,
                target_size=target_size,
                compute_loss=True,
                is_main_process=True,
            )
            
            # 保存最佳模型（基于 mAP）
            map_50_95 = val_results['map']
            if map_50_95 > best_map:
                is_best = True
                best_map = map_50_95
            else:
                is_best = False
        
        
        # 每个 epoch 保存 checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            output_dir=output_dir,
            is_best=is_best,
            is_main_process=True,
        )

if __name__ == "__main__":
    train()