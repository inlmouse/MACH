# dataset/unified_dataset.py
import json
import random
import os
import numpy as np
from typing import Any, Dict
from PIL import Image, ImageFile, ImageFont
import torch
from torch.utils.data import Dataset

# 允许加载截断的图片（尽可能加载能加载的部分）
ImageFile.LOAD_TRUNCATED_IMAGES = True

def check_cls_continuous(cls: torch.Tensor):
    """
    检查 cls 是否从 0 开始连续（例如 [0,1,2,3,...,K-1]）
    """
    cls = cls.view(-1).long()
    uniq = torch.unique(cls)
    uniq_sorted = torch.sort(uniq).values

    expected = torch.arange(len(uniq_sorted), device=cls.device)

    is_contiguous = torch.equal(uniq_sorted, expected)

    if not is_contiguous:
        missing = set(range(int(uniq_sorted.max().item()) + 1)) - set(uniq_sorted.tolist())
        print(f"❌ cls 不连续")
        print(f"   unique: {uniq_sorted.tolist()}")
        print(f"   missing: {sorted(list(missing))}")

    return is_contiguous

class UnifiedDetectionDataset(Dataset):
    def __init__(self, samples, label_list=None, num_infonce_batch=None, transforms=None, 
                 istrain=True, global_neg_cat_path='dataset/global_grounding_neg_cat.json'):
        """
        samples: List[UnifiedAnnotation]
        """
        self.samples = samples
        self.transforms = transforms
        self.label_list = label_list
        self.max_samples = num_infonce_batch if num_infonce_batch is not None else None
        self.istrain = istrain

        self.num_classes = len(label_list) if label_list is not None else 0

        with open(global_neg_cat_path, 'r', encoding='utf-8') as f:
            self.global_grounding_neg = json.load(f)
              

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        
        image = Image.open(s["file_name"]).convert("RGB")
        
        # 这里进行类别采样：如果 max_samples 不为 None 且是训练模式，就从原始标签中随机采样 max_samples 个类别（正负都采），并相应地过滤掉不在采样列表中的框和标签。同时，构建新的 captions 列表，包含采样的类别对应的 prompt，以及必要时的 padding。
        # 使用局部变量存储处理后的数据，避免修改原始 samples
        boxes = s["boxes"]
        labels = s["labels"]
        captions = s.get("captions", [])
        if self.max_samples is not None and self.istrain:
            old_cls = np.asarray(labels, dtype=int)
            
            # 检查是否所有 labels 都是 -1（未标注/grounding 模式）
            if np.all(old_cls == -1):
                # 从 captions 中取 unique，随机选择 max_samples 个（不足则全选）
                unique_captions = list(set(captions))
                if len(unique_captions) > self.max_samples:
                    selected_captions = random.sample(unique_captions, k=self.max_samples)
                else:
                    selected_captions = unique_captions
                
                # 构建 caption 到 id 的映射
                caption2id = {cap: i for i, cap in enumerate(selected_captions)}
                
                # 过滤 bbox：只保留被选中 caption 对应的框
                valid_idx = np.zeros(len(boxes), dtype=bool)
                new_cls = []
                for i, cap in enumerate(captions):
                    if cap in caption2id:
                        valid_idx[i] = True
                        new_cls.append([caption2id[cap]])
                
                boxes = np.array(boxes)[valid_idx]
                labels = np.array(new_cls)
                
                # 构建 texts 和 txt_feats（使用索引查询，批量 GPU 传输）
                texts = selected_captions.copy()

                # padding 到 max_samples
                num_padding = self.max_samples - len(texts)
                if num_padding > 0:
                    global_neg_cat_len = len(self.global_grounding_neg)
                    pad_net_cat_indexs = np.random.choice(np.arange(0, global_neg_cat_len), size=num_padding, replace=False)
                    pad_net_cat = [self.global_grounding_neg[i].strip() for i in pad_net_cat_indexs]
                    texts.extend(pad_net_cat)
                
                captions = texts
                
                # 构建 text_is_positive: 正样本为 True, padding 为 False
                # grounding 模式下 selected_captions 都是正样本，padding 是负样本
                text_is_positive = torch.tensor(np.array(
                    [True] * len(selected_captions) +
                    [False] * num_padding,
                    dtype=bool
                ))
            else:
                # 原有逻辑：labels 是正常赋值的情况
                pos_labels = np.unique(old_cls).tolist()
                if len(pos_labels) > self.max_samples:
                    pos_labels = random.sample(pos_labels, k=self.max_samples)

                neg_samples = min(min(self.num_classes, self.max_samples) - len(pos_labels), random.randint(self.max_samples, self.max_samples))
                neg_labels = [i for i in range(self.num_classes) if i not in pos_labels]
                neg_labels = random.sample(neg_labels, k=neg_samples)

                sampled_labels = pos_labels + neg_labels
                
                label2ids = {label: i for i, label in enumerate(sampled_labels)}
                valid_idx = np.zeros(len(boxes), dtype=bool)
                new_cls = []
                for i, label in enumerate(old_cls.tolist()):
                    if label not in label2ids:
                        continue
                    valid_idx[i] = True
                    new_cls.append([label2ids[label]])
                boxes = np.array(boxes)[valid_idx]
                labels = np.array(new_cls)

                # Randomly select one prompt when there's more than one prompts
                texts = []
                prompt_format = "{}"  # 可以根据需要修改这个格式，例如 "A photo of a {}." 或其他更复杂的模板. 这里先不动它，直接使用原始文本。
                for label in sampled_labels:
                    prompts = self.label_list[label]
                    assert len(prompts) > 0
                    prompt = prompt_format.format(prompts)
                    texts.append(prompt.strip())

                #padding 这里可以做hard example mining，找出topk个最相似的负样本
                valid_labels = len(pos_labels) + len(neg_labels)
                num_padding = self.max_samples - valid_labels
                if num_padding > 0:
                    global_neg_cat_len = len(self.global_grounding_neg)
                    pad_net_cat_indexs = np.random.choice(np.arange(0, global_neg_cat_len), size=num_padding, replace=False)
                    pad_net_cat = [self.global_grounding_neg[i].strip() for i in pad_net_cat_indexs]
                    texts.extend(pad_net_cat)
                captions = texts
                
                # 构建 text_is_positive: 正样本为 True, 负样本和 padding 为 False
                text_is_positive = torch.tensor(np.array(
                    [label in pos_labels for label in sampled_labels] +
                    [False] * num_padding,
                    dtype=bool
                ))
        #check_cls_continuous(torch.as_tensor(labels))
        target: Dict[str, Any] = {
                "image_id": torch.tensor(s["image_id"]),
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "captions": captions,                 # List[str]
                # "segmentations": s["segmentations"],       # List[Any]
                "text_is_positive": text_is_positive,  # [num_infonce_batch] bool
                "allow_aug": s.get("allow_aug", False),
            }

        if self.transforms is not None:
           image, target = self.transforms(image, target)


        # ──────────────── Debug: 检查 transform 后的坐标 ────────────────
        DEBUG = False   # 改成 True 开启
        if DEBUG and len(target["boxes"]) > 0:
            try:
                from PIL import ImageDraw
                import torchvision.transforms as T

                # tensor [C,H,W] -> PIL Image
                img_tensor = image.clone()
                # 反归一化（假设是标准 ImageNet 归一化）
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
                img_tensor = img_tensor.clamp(0, 1)
                
                # 转为 PIL
                to_pil = T.ToPILImage()
                draw_img = to_pil(img_tensor)
                draw = ImageDraw.Draw(draw_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
                except Exception:
                    font = ImageFont.load_default()

                boxes = target["boxes"].cpu().numpy()   # [N, 4] cxcywh ([0, 1])
                labels = target["labels"].cpu().numpy() # [N]
                h, w = image.shape[1], image.shape[2]   # tensor shape [C,H,W]

                for i, box in enumerate(boxes):
                    if len(box) != 4:
                        continue
                    
                    # transform 后的坐标是归一化的 cxcywh ([0, 1])
                    # 需要转换为 xyxy 并乘以图像尺寸
                    cx, cy, bw, bh = box
                    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                    
                    x1 = int(cx - bw / 2)
                    y1 = int(cy - bh / 2)
                    x2 = int(cx + bw / 2)
                    y2 = int(cy + bh / 2)
                    
                    # 限制在图像范围内
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # 画框（红色）
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                    # 标签
                    label_str = f"cls{labels[i]}"
                    if "captions" in target:
                        label_idx = labels[i]
                        if isinstance(label_idx, (list, np.ndarray)):
                            label_idx = label_idx[0]
                        cap_idx = int(label_idx)
                        if 0 <= cap_idx < len(target["captions"]):
                            cap = target["captions"][cap_idx]
                            label_str += f" {str(cap)[:20]}"
                    
                    draw.text((x1, max(0, y1-25)), label_str, fill="red", font=font)

                # 保存
                save_path = f"debug_transform_{idx:06d}_id{target['image_id'].item()}.jpg"
                draw_img.save(save_path)
                print(f"Debug (transform 后): {save_path} | {len(boxes)} 个框 | 图像大小: {w}x{h}")

            except Exception as e:
                print(f"Debug 绘图失败: {e}")
                import traceback
                traceback.print_exc()
        # ──────────────── Debug 结束 ────────────────
        return image, target
