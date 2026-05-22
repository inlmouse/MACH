import json
from pathlib import Path
import torch
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visualization_utils import draw_predictions

from dataset.transforms import make_coco_transforms
from utils.detect_utils import non_max_suppression, scale_boxes
from utils.train_utils import load_model
from dataset.textmodelembedder import Qwen3VLEmbeddingTextEmbedder, CLIPTextEmbedder

TEXT_EMBED_DIM = 768
TARGET_SIZE = 640
DEVICE = "cuda:0"
QWEN_MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-Embedding-2B"
CKPT_PATH = "outputs-qwen2b-768/model_mgfixed_vitneck_woffn_tiny_epoch30.pth"

CONF_THRESH = 0.05
NMS_THRESH = 0.75

def box_iou(boxes1, boxes2):
        area1 = (boxes1[:,2]-boxes1[:,0]).clamp(0) * (boxes1[:,3]-boxes1[:,1]).clamp(0)
        area2 = (boxes2[:,2]-boxes2[:,0]) * (boxes2[:,3]-boxes2[:,1])

        lt = torch.max(boxes1[:,:2], boxes2[:,:2])
        rb = torch.min(boxes1[:,2:], boxes2[:,2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:,0]*wh[:,1]
        union = area1 + area2 - inter
        return inter/union.clamp(min=1e-6)

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

def load_gt_dict(coco_json_path, model=None, textencoder=None):
    with open(coco_json_path,'r') as f:
        coco = json.load(f)
    img_dict = {img['id']: img for img in coco['images']}
    gt_dict = {}
    image_to_captions = defaultdict(list)
    tp_counter = 0
    valid_count = 0
    for ann in tqdm(coco['annotations']):
        if ann['caption_quality'] <= 0:
            continue
        valid_count += 1
        image_id = ann['image_id']
        bbox = ann['bbox']
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]  # 转为 [x1, y1, x2, y2]
        caption = img_dict[image_id]['caption']
        dataset_name = ann.get('dataset_name','all')
        file_name = img_dict[image_id]['file_name'] 
        gt_dict[image_id] = {"bbox": bbox, "dataset_name": dataset_name, "caption": caption, "file_name": file_name}
        image_to_captions[caption].append(image_id)
        textfeats = textencoder.embedtext([caption], normalize=True).to(DEVICE)
        img_path = "/root/autodl-tmp/OOD/refcoco/images/train2014/"+file_name
        model.set_class(textfeats)
        image, boxes, scores, labels = inference_single(
                model, str(img_path), textfeats, DEVICE, TARGET_SIZE, 
                CONF_THRESH, NMS_THRESH
            )
        model.unset_class()
        if hasattr(boxes, 'cpu'):
            boxes = boxes.cpu()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu()
        if hasattr(labels, 'cpu'):
            labels = labels.cpu()
        
        
        if len(scores) <= 0:
            continue
        best_idx = np.argmax(scores)
        best_box = boxes[best_idx][None, :]   # (1, 4)
        bboxGT = torch.tensor([bbox])
        iou = box_iou(best_box, bboxGT)
        if iou >= 0.5:
            tp_counter += 1
            # boxes = np.asarray(best_box)
            # scores = np.asarray(scores[best_idx])
            # labels = np.asarray(labels[best_idx], dtype=int)
            # all_boxes = np.vstack([[bbox], boxes])
            # all_scores = np.hstack(([1.0], scores)) 
            # all_labels = np.hstack(([1], labels))
            # all_texts = [caption]+[caption + "(GT)"]
            # #vis_image = draw_predictions(image, [bbox], [1.0], [0], [caption+"(GT)"])
            # vis_image = draw_predictions(image, boxes, [scores], [labels], [caption])
            # #vis_image = draw_predictions(image, all_boxes, all_scores, all_labels, all_texts)
            # image.save("/root/autodl-tmp/VLMs/test_results/"+file_name)
            # a = 1
        # else:
        #     boxes = np.asarray(boxes)
        #     scores = np.asarray(scores)
        #     labels = np.asarray(labels, dtype=int)
        #     all_boxes = np.vstack([[bbox], boxes])
        #     all_scores = np.hstack(([1.0], scores)) 
        #     all_labels = np.hstack(([1], labels))
        #     all_texts = [caption]+[caption + "(GT)"]
        #     vis_image = draw_predictions(image, [bbox], [1.0], [0], [caption+"(GT)"])
        #     vis_image = draw_predictions(vis_image, boxes, scores, labels, [caption])
        #     vis_image = draw_predictions(image, all_boxes, all_scores, all_labels, all_texts)
        #     vis_image.save("/root/autodl-tmp/VLMs/test_results/"+file_name)
        #     a = 1
    acc = tp_counter * 1.0 / valid_count
    print(f"Dataset {coco_json_path}, Accuracy: {acc:.4f}")
    return gt_dict, coco['images'], image_to_captions


if __name__ == "__main__":
    textencoder = Qwen3VLEmbeddingTextEmbedder(
            QWEN_MODEL_PATH,
            device=DEVICE,
            mrl_truncate=TEXT_EMBED_DIM
    )
    model, _, _ = load_model(CKPT_PATH, TEXT_EMBED_DIM, None, DEVICE)

    # load_gt_dict("/root/autodl-tmp/OOD/refcoco/annotations/refcoco_val_reviewed.json", model, textencoder)
    load_gt_dict("/root/autodl-tmp/OOD/refcoco/annotations/refcoco_testA_reviewed.json", model, textencoder)
    # load_gt_dict("/root/autodl-tmp/OOD/refcoco/annotations/refcoco_testB_reviewed.json", model, textencoder)
    # load_gt_dict("/root/autodl-tmp/OOD/refcoco/annotations/refcoco+_val_reviewed.json", model, textencoder)
    # load_gt_dict("/root/autodl-tmp/OOD/refcoco/annotations/refcoco+_testA_reviewed.json", model, textencoder)
    # load_gt_dict("/root/autodl-tmp/OOD/refcoco/annotations/refcoco+_testB_reviewed.json", model, textencoder)
    # load_gt_dict("/root/autodl-tmp/OOD/refcoco/annotations/refcocog_val_reviewed.json", model, textencoder)
    # load_gt_dict("/root/autodl-tmp/OOD/refcoco/annotations/refcocog_test_reviewed.json", model, textencoder)

