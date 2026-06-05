# 检测模型测试 + 可视化脚本

import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from models.expalignet import expalignet
from dataset.transforms import make_coco_transforms
from utils.detect_utils import non_max_suppression, scale_boxes, tensor2img
from utils.cnn_train_utils import load_model
from utils.visualization_utils import draw_predictions
from dataset.build_dataloader import load_labels
from dataset.textmodelembedder import Qwen3VLEmbeddingTextEmbedder, CLIPTextEmbedder


# ==================== 配置区域 ====================
# 根据你的测试需求修改以下配置

CKPT_PATH = "outputs-qwen2b-768/model_film_woffn_tiny_epoch30.pth"           # 模型检查点路径
IMAGE_PATH = "/root/autodl-tmp/OOD/coco/images/val2017/000000367569.jpg"  # 单张图片路径
IMAGE_PATH = "/root/autodl-tmp/VLMs/test_results/personinmirror.jpg"
# IMAGE_PATH = ["/root/autodl-tmp/VLMs/test_results/test2.jpg",
#               "/root/autodl-tmp/VLMs/test_results/test3.jpg"]
IMAGE_DIR = None                                          # 图片目录（与 IMAGE_PATH 二选一）
OUTPUT_DIR = "test_results"                               # 可视化结果保存目录

# 模型配置
TEXT_EMBED_DIM = 768
TARGET_SIZE = 640
DEVICE = "cuda:3"

# 推理配置
CONF_THRESH = 0.01
NMS_THRESH = 0.75
MAX_IMAGES = None  # 最多处理多少张图片，None 表示处理全部

# 类别配置（根据你的测试数据修改）
CLASS_NAMES = [
    'bottle', 'the picture on the wall', 'table', 'fireplace', 'stool', 'electronic piano',
    'curtain', 'APPLE laptop', 'remote control', 'pen', 'open book', 'lamp on ceiling', 'pillow'
]

CLASS_NAMES = [
    'shoes', 'black knee-high socks', 'pleated skirt', 'sailor uniform', 'school bag', 'person', 'bangs', 'pony tail', 'rope', 'bench'
]
CLASS_NAMES = ['person in mirror', 'hair dryer']
# 文本编码器配置
TEXT_ENCODER_TYPE = "qwen"  # "qwen" 或 "clip"
QWEN_MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-Embedding-2B"
CLIP_MODEL_NAME = "/project/GLS/HJY/yoloe/ViT-L-14.pt"

# ==================== 功能函数 ====================

def preprocess_image(image_path, target_size=640):
    """预处理单张图片"""
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    
    transforms = make_coco_transforms(istrain=False, target_size=target_size)
    image_tensor, _ = transforms(image, None)
    #img = tensor2img(image_tensor)  
    #img.save("/project/GLS/HJY/VLMs/test_results/test-preinfer.jpg")
    return image_tensor, (orig_w, orig_h), image


def inference_single(model, image_path, text_feats=None, device='cuda', target_size=640, 
                     conf_thresh=0.25, nms_thresh=0.5):
    """单张图片推理"""
    # 预处理
    image_tensor, orig_size, orig_image = preprocess_image(image_path, target_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    orig_w, orig_h = orig_size
    
    # 推理
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(image_tensor, text_feats)
            pred = outputs[0] if isinstance(outputs, tuple) else outputs
    
    # NMS 后处理
    preds = non_max_suppression(pred, conf_thres=conf_thresh, iou_thres=nms_thresh)
    pred = preds[0]
    
    if len(pred) == 0:
        return orig_image, torch.empty(0, 4), torch.empty(0), torch.empty(0)
    
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
    
    return orig_image, boxes, scores, labels


def get_test_images(image_path, image_dir, max_images=None):
    """获取测试图片列表"""
    
    if isinstance(image_path, list):
        return image_path[:max_images] if max_images else image_path
    if isinstance(image_path, str):
        return [image_path]
    if image_dir:
        image_paths = (
            list(Path(image_dir).glob('*.jpg')) +
            list(Path(image_dir).glob('*.png')) +
            list(Path(image_dir).glob('*.jpeg'))
        )
        if max_images:
            image_paths = image_paths[:max_images]
        return image_paths
    
    raise ValueError("请配置 IMAGE_PATH 或 IMAGE_DIR")


def build_text_encoder(encoder_type, embed_dim, device):
    """构建文本编码器"""
    if encoder_type == "qwen":
        return Qwen3VLEmbeddingTextEmbedder(
            QWEN_MODEL_PATH, 
            device=device, 
            mrl_truncate=embed_dim
        )
    elif encoder_type == "clip":
        return CLIPTextEmbedder(
            CLIP_MODEL_NAME, 
            device=device, 
            mrl_truncate=embed_dim
        )
    else:
        raise ValueError(f"不支持的文本编码器类型: {encoder_type}")


# ==================== 主函数 ====================

def main():
    # 1. 准备输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. 加载模型
    print(f"加载模型: {CKPT_PATH}")
    model, _, _ = load_model(CKPT_PATH, TEXT_EMBED_DIM, None, DEVICE)
    print(f"模型加载完成，类别数: {len(CLASS_NAMES)}，设备: {DEVICE}")
    
    # 3. 加载文本编码器并设置类别
    text_encoder = build_text_encoder(TEXT_ENCODER_TYPE, TEXT_EMBED_DIM, DEVICE)
    
    # 4. 获取测试图片
    image_paths = get_test_images(IMAGE_PATH, IMAGE_DIR, MAX_IMAGES)
    print(f"找到 {len(image_paths)} 张测试图片")
    
    # 5. 逐张推理
    for img_path in tqdm(image_paths, desc="推理中"):
        try:
            
            text_feats = text_encoder.embedtext(CLASS_NAMES, normalize=True)
            model.set_class(text_feats)
            image, boxes, scores, labels = inference_single(
                model, str(img_path), text_feats, DEVICE, TARGET_SIZE, 
                CONF_THRESH, NMS_THRESH
            )
            model.unset_class()
            print(f"{img_path}: 检测到 {len(boxes)} 个目标")
            
            # 可视化
            vis_image = draw_predictions(
                image, boxes, scores, labels, 
                class_names=CLASS_NAMES,
            )
            
            # 保存
            save_name = Path(img_path).stem + '_result.jpg'
            save_path = os.path.join(OUTPUT_DIR, save_name)
            vis_image.save(save_path)
            
        except Exception as e:
            print(f"处理 {img_path} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"结果已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
