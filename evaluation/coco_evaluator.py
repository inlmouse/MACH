# coco_evaluator.py
# 目标检测评估指标计算（mAP 等）
import numpy as np
import torch
from collections import defaultdict

class DetectionEvaluator:
    """
    目标检测评估器（基于 faster-coco-eval v1.7+）
    - 完全修复 COCO.fromjson 不存在的问题
    - 直接支持 in-memory dict（无需写 JSON 文件）
    - 接口与之前完全一致
    """
    
    def __init__(self, num_classes, iou_threshes=None, fast_mode=False):
        self.num_classes = num_classes
        self.fast_mode = fast_mode
        
        # COCO 格式数据收集
        self.gt_anns = []
        self.pred_anns = []
        self.unique_img_ids = set()
        self.ann_counter = 0
        self.categories = [{"id": i, "name": f"class_{i}"} for i in range(num_classes)]
    
    def add_batch(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, img_ids, gt_areas=None):
        """添加一批结果（接口完全不变）"""
        for i, img_id in enumerate(img_ids):
            self.unique_img_ids.add(img_id)
            
            # ====================== GT ======================
            if len(gt_boxes[i]) > 0:
                boxes = gt_boxes[i]          # (K, 4) xyxy
                labels = gt_labels[i]
                areas = gt_areas[i] if gt_areas is not None else \
                        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                
                for j in range(len(labels)):
                    x1, y1, x2, y2 = boxes[j].tolist()
                    gt_ann = {
                        "id": self.ann_counter,
                        "image_id": img_id,
                        "category_id": int(labels[j].item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],   # COCO 需要 xywh
                        "area": float(areas[j].item()),
                        "iscrowd": 0,
                    }
                    self.gt_anns.append(gt_ann)
                    self.ann_counter += 1
            
            # ====================== Predictions ======================
            if len(pred_boxes[i]) > 0:
                boxes = pred_boxes[i]
                scores = pred_scores[i]
                labels = pred_labels[i]
                for j in range(len(scores)):
                    x1, y1, x2, y2 = boxes[j].tolist()
                    pred_ann = {
                        "image_id": img_id,
                        "category_id": int(labels[j].item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(scores[j].item()),
                    }
                    self.pred_anns.append(pred_ann)
    
    def compute_map(self, conf_thresh=0.001):
        """计算 mAP（已修复）"""
        if not self.pred_anns or not self.gt_anns:
            return self._empty_results()
        
        # 过滤低置信度
        filtered_preds = [p for p in self.pred_anns if p["score"] >= conf_thresh]
        if not filtered_preds:
            return self._empty_results()
        
        # 构建 COCO 格式 dict（in-memory）
        images = [{"id": iid} for iid in sorted(self.unique_img_ids)]
        anno_dict = {
            "images": images,
            "annotations": self.gt_anns,
            "categories": self.categories,
        }
        
        # ====================== 关键修复点 ======================
        from faster_coco_eval import COCO, COCOeval_faster
        
        anno = COCO(anno_dict)                    # ← 直接传入 dict（不再用 fromjson）
        pred = anno.loadRes(filtered_preds)       # ← 传入 list（不是 dict）
        
        evaluator = COCOeval_faster(anno, pred, iouType="bbox")
        
        # fast_mode 只算 IoU=0.5
        if self.fast_mode:
            evaluator.params.iouThrs = np.array([0.5])
        
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        
        stats = evaluator.stats_as_dict
        
        results = {
            "mAP@0.5": stats.get("AP_50", 0.0),
            "mAP@0.5:0.95": stats.get("AP_all", 0.0),
            "AP_S@0.5": stats.get("AP_small", 0.0),
            "AP_M@0.5": stats.get("AP_medium", 0.0),
            "AP_L@0.5": stats.get("AP_large", 0.0),
        }
        
        if not self.fast_mode and "AP_75" in stats:
            results["mAP@0.75"] = stats["AP_75"]
        
        return results
    
    def _empty_results(self):
        return {
            "mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0,
            "AP_S@0.5": 0.0, "AP_M@0.5": 0.0, "AP_L@0.5": 0.0,
        }
    
    def reset(self):
        """清空数据"""
        self.gt_anns.clear()
        self.pred_anns.clear()
        self.unique_img_ids.clear()
        self.ann_counter = 0


def print_map_results(results):
    """打印 mAP 结果（保持不变）"""
    print("\n" + "=" * 60)
    print("验证结果 (COCO 标准)")
    print("=" * 60)
    print(f"mAP@0.5       (IoU=0.50):      {results.get('mAP@0.5', 0.0):.4f}")
    if 'mAP@0.75' in results:
        print(f"mAP@0.75      (IoU=0.75):      {results['mAP@0.75']:.4f}")
    if 'mAP@0.5:0.95' in results:
        print(f"mAP@0.5:0.95  (IoU=0.50:0.95): {results['mAP@0.5:0.95']:.4f}")
    print("-" * 60)
    print("目标大小 (IoU=0.50):")
    print(f"  AP_S (Small):  area < 32²     {results.get('AP_S@0.5', 0.0):.4f}")
    print(f"  AP_M (Medium): 32² <= area < 96² {results.get('AP_M@0.5', 0.0):.4f}")
    print(f"  AP_L (Large):  area >= 96²    {results.get('AP_L@0.5', 0.0):.4f}")
    print("=" * 60 + "\n")