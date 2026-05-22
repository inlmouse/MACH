#!/usr/bin/env python3
"""
LVIS 离线评估脚本
使用 lvis 库对训练好的模型在 LVIS 数据集上进行评估

用法:
    python evaluation/lvis_offline_evaluator.py \
        --checkpoint outputs/model_last.pth \
        --lvis_json /path/to/lvis_v1_val.json \
        --image_root /path/to/lvis/val2017 \
        --text_encoder clip \
        --batch_size 8 \
        --device cuda
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.expalignet import expalignet
from dataset.transforms import make_coco_transforms
from utils.detect_utils import non_max_suppression, scale_boxes
from utils.train_utils import load_model
from utils.visualization_utils import draw_predictions
from dataset.textmodelembedder import Qwen3VLEmbeddingTextEmbedder, CLIPTextEmbedder


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LVIS 离线评估")
    
    # 模型配置
    parser.add_argument("--checkpoint", type=str, default="outputs-qwen2b-768/model_last.pth", 
                        help="模型检查点路径 (.pth)")
    parser.add_argument("--text_embed_dim", type=int, default=768,
                        help="文本嵌入维度")
    parser.add_argument("--target_size", type=int, default=640,
                        help="图像目标尺寸")
    parser.add_argument("--device", type=str, default="cuda:5",
                        help="计算设备 (cuda 或 cpu)")
    
    # 数据配置
    parser.add_argument("--lvis_json", type=str, default="/root/autodl-tmp/OOD/coco/annotations/lvis_v1_minival.json",
                        help="LVIS 标注文件路径 (如 lvis_v1_val.json)")
    parser.add_argument("--image_root", type=str, default="/root/autodl-tmp/OOD/coco/images",
                        help="图像根目录路径")
    
    # 文本编码器配置
    parser.add_argument("--text_encoder", type=str, default="qwen",
                        choices=["qwen", "clip"],
                        help="文本编码器类型")
    parser.add_argument("--qwen_model_path", type=str,
                        default="/root/autodl-tmp/Qwen3-VL-Embedding-2B",
                        help="Qwen 模型路径 (当 text_encoder=qwen 时使用)")
    parser.add_argument("--clip_model_name", type=str,
                        default="/root/autodl-tmp/yoloe/ViT-L-14.pt",
                        help="CLIP 模型路径 (当 text_encoder=clip 时使用)")
    
    # 推理配置
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批处理大小")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="数据加载线程数")
    parser.add_argument("--conf_thresh", type=float, default=0.001,
                        help="置信度阈值")
    parser.add_argument("--nms_thresh", type=float, default=0.75,
                        help="NMS 阈值")
    
    # 评估配置
    parser.add_argument("--max_images", type=int, default=None,
                        help="最大评估图像数 (用于调试)")
    parser.add_argument("--split", type=str, default="minival", choices=["val", "minival"],
                        help="评估数据集划分: val (完整验证集) 或 minival (5000张子集)")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="保存预测结果到 JSON 文件路径")
    
    return parser.parse_args()


class LVISDataset(Dataset):
    """
    LVIS 数据集封装（支持 val 和 minival）
    
    使用方式：
        # val
        dataset = LVISDataset(lvis_json='annotations/lvis_v1_val.json',
                              image_root='images/val2017',
                              split='val')
        
        # minival（推荐直接传入官方 minival json）
        dataset = LVISDataset(lvis_json='annotations/lvis_v1_minival.json',   # 或 lvis_v1_minival_inserted_image_name.json
                              image_root='images/val2017',
                              split='minival')
    """

    def __init__(self, lvis_json, image_root, target_size=640, max_images=None, split='val'):
        self.image_root = Path(image_root)
        self.target_size = target_size
        self.split = split.lower()

        # 加载 LVIS 标注文件
        print(f"加载 LVIS 标注: {lvis_json}")
        with open(lvis_json, 'r', encoding='utf-8') as f:
            self.lvis_data = json.load(f)

        # 类别信息
        self.categories = {cat['id']: cat for cat in self.lvis_data['categories']}
        self.category_names = [cat['name'] for cat in self.lvis_data['categories']]

        # 图像信息
        self.images = {img['id']: img for img in self.lvis_data['images']}
        self.image_ids = list(self.images.keys())

        # 标注映射: image_id -> list of annotations
        self.img_to_anns = defaultdict(list)
        for ann in self.lvis_data['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)

        # 根据 split 打印信息（不再手动过滤）
        num_images = len(self.image_ids)
        if self.split == 'minival':
            if num_images > 6000:
                print(f"警告: split='minival' 但加载的 JSON 包含 {num_images} 张图像（远多于预期 ~5000）。"
                      f"请确认是否传入了 lvis_v1_minival.json 而非 val.json")
            else:
                print(f"使用 minival 划分: {num_images} 张图像（JSON 已正确过滤）")
        else:  # val
            print(f"使用 val 划分: {num_images} 张图像")

        # 可选：限制最大图像数量（用于调试）
        if max_images is not None:
            self.image_ids = self.image_ids[:max_images]
            print(f"限制为前 {max_images} 张图像用于调试")

        # 数据集统计
        total_anns = sum(len(self.img_to_anns[iid]) for iid in self.image_ids)
        print(f"数据集加载完成 → {len(self.image_ids)} 张图像, "
              f"{len(self.categories)} 个类别, {total_anns} 个标注")

        # 预处理变换（验证模式）
        self.transforms = make_coco_transforms(istrain=False, target_size=target_size)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # 加载图像
        img_path = self.image_root / img_info['file_name']
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            # 返回空白图像
            image = Image.new('RGB', (img_info['width'], img_info['height']), (114, 114, 114))
        
        orig_w, orig_h = image.size
        
        # 预处理
        image_tensor, _ = self.transforms(image, None)
        
        # 构建目标信息
        anns = self.img_to_anns.get(img_id, [])
        gt_boxes = []
        gt_labels = []
        gt_areas = []
        
        for ann in anns:
            # LVIS bbox 格式: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # 转换为 [x1, y1, x2, y2]
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])
            gt_areas.append(ann.get('area', w * h))
        #draw_predictions(image, gt_boxes, [1.0 for _ in gt_labels], gt_labels, class_names=self.category_names).save(f"/project/GLS/HJY/VLMs/test_results/test-{img_id}-gt.jpg")
        target = {
            'image_id': img_id,
            'file_name': img_info['file_name'],
            'orig_size': (orig_h, orig_w),
            'gt_boxes': torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.empty(0, 4),
            'gt_labels': torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.empty(0),
            'gt_areas': torch.tensor(gt_areas, dtype=torch.float32) if gt_areas else torch.empty(0),
        }
        
        return image_tensor, target


def collate_fn(batch):
    """自定义 collate 函数"""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets


def build_text_encoder(encoder_type, embed_dim, device, args):
    """构建文本编码器"""
    if encoder_type == "qwen":
        return Qwen3VLEmbeddingTextEmbedder(
            args.qwen_model_path,
            device=device,
            mrl_truncate=embed_dim
        )
    elif encoder_type == "clip":
        return CLIPTextEmbedder(
            args.clip_model_name,
            device=device,
            mrl_truncate=embed_dim
        )
    else:
        raise ValueError(f"不支持的文本编码器类型: {encoder_type}")


def run_inference(model, dataloader, device, conf_thresh, nms_thresh, target_size):
    """
    运行推理，收集预测结果
    注意：模型需要提前调用 set_class() 设置类别
    
    返回:
        predictions: 列表，每个元素是 (img_id, boxes, scores, labels, orig_size)
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_images, batch_targets in tqdm(dataloader, desc="推理"):
            batch_images = batch_images.to(device)
            batch_size = batch_images.size(0)
            
            # 前向传播（不需要传入 text_feats，已通过 set_class 设置）
            with torch.cuda.amp.autocast():
                outputs = model(batch_images)
            
            # 对每个样本进行 NMS 后处理
            for i in range(batch_size):
                target = batch_targets[i]
                img_id = target['image_id']
                orig_h, orig_w = target['orig_size']
                
                # 获取预测结果
                pred = outputs[0][i] if isinstance(outputs, tuple) else outputs[i]
                
                # NMS
                preds = non_max_suppression(
                    pred.unsqueeze(0),
                    conf_thres=conf_thresh,
                    iou_thres=nms_thresh
                )
                pred = preds[0]
                
                if len(pred) == 0:
                    predictions.append((img_id, torch.empty(0, 4), torch.empty(0), torch.empty(0), (orig_h, orig_w)))
                    continue
                
                # 分离 box, score, label
                boxes = pred[:, :4]
                scores = pred[:, 4]
                labels = pred[:, 5]
                
                # 将坐标从 target_size 空间映射回原图空间
                boxes = scale_boxes(
                    img1_shape=(target_size, target_size),
                    boxes=boxes,
                    img0_shape=(orig_h, orig_w)
                )
                #draw_predictions(
                #    Image.open(Path(dataloader.dataset.image_root) / target['file_name']).convert("RGB"),
                #    boxes.cpu().numpy(),
                #    scores.cpu().numpy(),
                #    labels.cpu().numpy(),
                #    class_names=dataloader.dataset.category_names
                #).save(f"/project/GLS/HJY/VLMs/test_results/test-{img_id}-pred.jpg")           
                predictions.append((img_id, boxes.cpu(), scores.cpu(), labels.cpu(), (orig_h, orig_w), target['gt_labels']))
    
    return predictions


def convert_to_lvis_format(predictions, dataset):
    """
    将预测结果转换为 LVIS 评估格式
    
    返回:
        results: 列表，每个元素是 LVIS 格式的检测结果字典
    """
    results = []

    # 创建 模型label_idx → category_id 的映射表（推荐方式）
    label_to_catid = {}
    for cat_id, cat_info in dataset.categories.items():
        # 如果你的模型类别列表和 LVIS categories 顺序一致（最常见情况）
        try:
            label_idx = dataset.category_names.index(cat_info['name'])
            label_to_catid[label_idx] = cat_id
        except ValueError:
            continue  # 该类别不在当前数据集子集中
    
    for img_id, boxes, scores, labels, _, gt_labels in predictions:
        # 将模型输出的 label 索引映射回 LVIS category_id
        # 模型输出的是 0-based 索引，需要映射到实际的 category_id
        for box, score, label in zip(boxes, scores, labels):
            # 获取类别名称
            label_idx = int(label.item())
            # 获取正确的 LVIS category_id
            if label_idx in label_to_catid:
                category_id = label_to_catid[label_idx]
            else:
                continue  # 跳过未知类别
            
            x1, y1, x2, y2 = box.tolist()
            # LVIS 格式: [x, y, width, height]
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            if int(category_id) in gt_labels:
                results.append({
                    'image_id': int(img_id),
                    'category_id': int(category_id),
                    'bbox': bbox,
                    'score': float(score.item()),
                })
    
    return results


def evaluate_lvis(predictions_json_path, lvis_gt_path):
    """
    使用 lvis 库进行评估
    
    Args:
        predictions_json_path: 预测结果 JSON 文件路径
        lvis_gt_path: LVIS 标注文件路径
    """
    try:
        from lvis import LVIS, LVISResults, LVISEval
    except ImportError:
        print("错误: 未安装 lvis 库，请运行: pip install lvis")
        print("官方仓库: https://github.com/lvis-dataset/lvis-api")
        sys.exit(1)
    
    print(f"\n使用 LVIS 库进行评估...")
    print(f"GT: {lvis_gt_path}")
    print(f"预测: {predictions_json_path}")
    
    # 加载 GT
    lvis_gt = LVIS(lvis_gt_path)
    
    # 加载预测结果
    lvis_results = LVISResults(lvis_gt, predictions_json_path)
    
    # 评估
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type='bbox')
    lvis_eval.run()
    lvis_eval.print_results()
    
    # 返回详细结果
    results = lvis_eval.get_results()
    return results


def main():
    args = parse_args()
    
    # 1. 加载数据集
    print("=" * 60)
    print("1. 加载 LVIS 数据集")
    print("=" * 60)
    dataset = LVISDataset(
        lvis_json=args.lvis_json,
        image_root=args.image_root,
        target_size=args.target_size,
        max_images=args.max_images,
        split=args.split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 2. 加载模型
    print("\n" + "=" * 60)
    print("2. 加载模型")
    print("=" * 60)
    model, _, _ = load_model(args.checkpoint, args.text_embed_dim, None, args.device)
    model.eval()
    print(f"模型加载完成: {args.checkpoint}")
    
    # 3. 构建文本编码器
    print("\n" + "=" * 60)
    print("3. 构建文本编码器")
    print("=" * 60)
    text_encoder = build_text_encoder(
        args.text_encoder,
        args.text_embed_dim,
        args.device,
        args
    )
    print(f"文本编码器加载完成，类别数: {len(dataset.category_names)}")
    
    # 4. 设置类别（open-vocabulary 模型需要先 set_class）
    print("\n" + "=" * 60)
    print("4. 设置模型类别")
    print("=" * 60)
    print(f"编码 {len(dataset.category_names)} 个 LVIS 类别...")
    with torch.no_grad():
        text_feats = text_encoder.embedtext(dataset.category_names, normalize=True)
        text_feats = text_feats.to(args.device)
        model.set_class(text_feats)
    print(f"类别设置完成，文本特征形状: {text_feats.shape}")
    
    # 5. 运行推理
    print("\n" + "=" * 60)
    print("5. 运行推理")
    print("=" * 60)
    predictions = run_inference(
        model=model,
        dataloader=dataloader,
        device=args.device,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        target_size=args.target_size
    )
    
    # 6. 转换格式并保存
    print("\n" + "=" * 60)
    print("6. 转换预测结果格式")
    print("=" * 60)
    results = convert_to_lvis_format(predictions, dataset)
    print(f"总检测框数: {len(results)}")
    
    # 保存预测结果
    if args.save_predictions:
        with open(args.save_predictions, 'w') as f:
            json.dump(results, f)
        print(f"预测结果已保存: {args.save_predictions}")
    
    # 7. LVIS 评估
    print("\n" + "=" * 60)
    print("7. LVIS 评估")
    print("=" * 60)
    
    # 临时保存预测结果用于评估
    temp_pred_path = args.save_predictions or f'/tmp/lvis_{args.split}_predictions.json'
    if not args.save_predictions:
        with open(temp_pred_path, 'w') as f:
            json.dump(results, f)
    
    # 运行评估
    eval_results = evaluate_lvis(temp_pred_path, args.lvis_json)
    
    # 清理临时文件
    if not args.save_predictions and os.path.exists(temp_pred_path):
        os.remove(temp_pred_path)
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
