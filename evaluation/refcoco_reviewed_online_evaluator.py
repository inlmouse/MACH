import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import inspect

# 延迟导入或直接导入你的 NMS 工具（供旧版模型使用）
from utils.detect_utils import non_max_suppression, scale_boxes

def box_iou(boxes1, boxes2):
    """
    计算两组 box 的 IoU。
    boxes1: [N, 4], boxes2: [M, 4]  -> 返回 [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])          
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])          
    wh = (rb - lt).clamp(min=0)                                 
    inter = wh[:, :, 0] * wh[:, :, 1]                           
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)

def preprocess_image(image_path, target_size=640):
    """预处理单张图片"""
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    from dataset.transforms import make_coco_transforms
    transforms = make_coco_transforms(istrain=False, target_size=target_size)
    image_tensor, _ = transforms(image, None)
    return image_tensor, (orig_w, orig_h), image


def inference_single(model, image_path, text_feats, mask=None, device='cuda',
                     target_size=640, conf_thresh=0.25, nms_thresh=0.5):
    """
    终极多态推理接口：
    依赖底层的 predict 封装，彻底消灭 if/else 架构路由。
    """
    image_tensor, orig_size, orig_image = preprocess_image(image_path, target_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    orig_w, orig_h = orig_size

    # 提取底层原始模型（剥离 DDP 包装）
    raw_model = model.module if hasattr(model, 'module') else model

    # 构造原图尺寸张量，交给底层做绝对坐标还原
    orig_target_sizes = torch.tensor([[orig_h, orig_w]], device=device)

    # ==========================================
    # 🚀 完美的鸭子类型调用 (Duck Typing)
    # 抹平 DETR 和 YOLO 在参数签名上的微小差异 (YOLO 需要 NMS)
    # ==========================================
    sig = inspect.signature(raw_model.predict)
    
    if 'nms_threshold' in sig.parameters:
        # 走 expalignet (YOLO) 路线
        results = raw_model.predict(
            x=image_tensor, 
            w=text_feats, 
            m=mask, 
            conf_threshold=conf_thresh, 
            nms_threshold=nms_thresh, 
            orig_target_sizes=orig_target_sizes
        )
    else:
        # 走 vlrtdetrnet (DETR) 路线
        results = raw_model.predict(
            x=image_tensor, 
            w=text_feats, 
            m=mask, 
            conf_threshold=conf_thresh, 
            orig_target_sizes=orig_target_sizes
        )

    # 统一输出格式，直接取第一张图的结果
    pred_dict = results[0]
    
    return orig_image, pred_dict["boxes"], pred_dict["scores"], pred_dict["labels"]


def validate_refcoco_one_epoch(
    model,
    device='cuda',
    epoch=0,
    target_size=640,
    textencoder=None,
    output_dir=None,
    is_main_process=True,
    coco_json_path=None,
    image_root=None,
    conf_thresh=0.01,
    nms_thresh=0.75,
):
    """
    纯粹的业务验证层：不关心模型架构，不关心模型状态，只负责评测指标。
    """
    if coco_json_path is None or not is_main_process:
        return {'refcoco_acc': 0.0, 'tp': 0, 'total': 0, 'epoch': epoch}

    # 1. 准备数据
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    img_dict = {img['id']: img for img in coco['images']}
    
    tp_counter = 0
    valid_count = 0

    # 2. 全局关闭梯度以节省显存
    with torch.no_grad():
        for ann in tqdm(coco['annotations'], desc=f"Epoch {epoch} RefCOCOg"):
            if ann.get('caption_quality', 1) <= 0:
                continue
            valid_count += 1

            # --- 解析标注 ---
            image_id = ann['image_id']
            bbox = ann['bbox']
            bbox_gt_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            img_info = img_dict[image_id]
            caption = img_info.get('caption', '')
            img_path = os.path.join(image_root, img_info['file_name'])

            # --- 文本特征提取 ---
            textfeats, mask = textencoder.embedtext([caption], normalize=True, batch_size=1, tokenlevel=True)
            textfeats = textfeats.to(device)
            mask = mask.to(device) if mask is not None else None

            # ==========================================
            # 🚀 极其黑盒的推理调用 (The Magic)
            # 你不再需要 model.eval()，不需要 set_class，不需要 NMS！
            # 哪怕你现在正处于训练的 step 中间，调它也绝对安全！
            # ==========================================
            _, boxes, scores, _ = inference_single(
                model=model, 
                image_path=str(img_path), 
                text_feats=textfeats, 
                mask=mask,
                device=device, 
                target_size=target_size,
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh
            )

            # --- 业务逻辑：算 IoU 并统计 True Positive ---
            if len(scores) > 0:
                best_idx = torch.argmax(scores)
                best_box = boxes[best_idx][None, :]  # 取出得分最高的框 [1, 4]

                bboxGT = torch.tensor([bbox_gt_xyxy], dtype=torch.float32, device=device)
                iou = box_iou(best_box, bboxGT)

                if iou.item() >= 0.5:
                    tp_counter += 1

    # 3. 汇总指标
    acc = tp_counter / valid_count if valid_count > 0 else 0.0
    
    # 获取底层网络名字用于日志展示
    raw_model = model.module if hasattr(model, 'module') else model
    arch_name = "VL-RT-DETR" if hasattr(raw_model, 'predict') else "ExpAlignNet"
    
    print(f"\n[{arch_name} - RefCOCOg] Epoch {epoch} | Accuracy@0.5: {acc:.4f} ({tp_counter}/{valid_count})")

    return {
        'refcoco_acc': acc,
        'tp': tp_counter,
        'total': valid_count,
        'epoch': epoch,
    }