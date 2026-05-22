# dataset/parsers/coco_parser.py
import os
from typing import List, Dict, Any
from tqdm import tqdm
from pycocotools.coco import COCO

from .base import BaseParser


class COCOParser(BaseParser):
    def __init__(self, ann_file: str, image_root: str, allow_complex_augmentation: bool = False):
        super().__init__(ann_file, image_root)
        self.coco = COCO(ann_file)
        self.ann_file = ann_file
        self.allow_complex_augmentation = allow_complex_augmentation
        # category_id -> name
        self.cat_id_to_name = {
            cid: cat["name"]
            for cid, cat in self.coco.cats.items()
        }

    def parse(self) -> List[Dict[str, Any]]:
        samples = []
        no_valid_count = 0
        img_ids = self.coco.getImgIds()
        for image_id in tqdm(img_ids, desc=f"Processing {len(img_ids)} images", unit="img"):
            img = self.coco.loadImgs(image_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            boxes = []
            labels = []         # 这里可以存 category_id（如果需要闭集兼容）
            captions = []       # 核心：用 phrase 或 fallback 到类别名
            captions_zh = []    # 可选：如果有中文 caption，可以单独存储
            segmentations = []

            for ann in anns:
                cid = ann["category_id"]
                # 处理 bounding box (xywh → 保持原样，或你可转 xyxy)
                #boxes.append(ann["bbox"])  # [x, y, w, h]
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])# [x, y, x, y]

                caption = None
                caption_zh = None
                # 优先级 1: ann 自己有独立的 caption（最直接、最可靠）
                if "caption" in ann and ann["caption"] and isinstance(ann["caption"], str):
                    caption = ann["caption"].lower().strip()
                    caption_zh = ann.get("caption_zh", None)
                    # 对于开集标签统一给-1
                    labels.append(-1)

                # 优先级 2: 有图像级 caption + tokens_positive（也即经典 grounding 方式）
                if caption is None:
                    has_imagelevel_caption = "caption" in img and img["caption"] is not None
                    # 用 tokens_positive 提取 phrase，否则 fallback
                    if has_imagelevel_caption and "tokens_positive" in ann and ann["tokens_positive"]:
                        try:
                            phrase_parts = [
                                img["caption"][t[0]:t[1]]
                                for t in ann["tokens_positive"]
                            ]
                            caption = " ".join(phrase_parts).lower().strip()
                            # 对于开集标签统一给-1
                            labels.append(-1)
                        except (IndexError, TypeError):
                            # 索引越界或格式异常 → fallback
                            continue
                            caption = self.cat_id_to_name.get(cid, "unknown")
                    else:
                        # 没有 caption / 没有 tokens_positive → 用标准类别名
                        caption = self.cat_id_to_name.get(cid, "unknown")
                        # 对于闭集标签先统一给-2，最后合并所有闭集数据集时再统一给label
                        labels.append(-2)

                captions.append(caption)
                captions_zh.append(caption_zh)

                # 4. segmentation（保持原样）
                segmentations.append(ann.get("segmentation", None))

            # 过滤掉没有有效标注框的样本
            if len(boxes) == 0:
                no_valid_count += 1
                continue
            if "file_name" in img:
                filename = img["file_name"]
            elif "coco_url" in img: #兼容lvis
                filename = img['coco_url'].strip().split('/')[-1]
            else:
                print(f"Warning: Image {image_id} has no valid file name or URL, skipping.")
                continue

            samples.append({
                "image_id": image_id,
                "file_name": os.path.join(self.image_root, filename),
                "height": img["height"],
                "width": img["width"],
                "boxes": boxes,             # List[List[float]]，每个是 [x,y,w,h]
                "labels": labels,           # List[int]，category_id
                "captions": captions,       # List[str]，现在是 phrase 或类别名
                "captions_zh": captions_zh, # List[str]，中文 caption
                # "segmentations": segmentations,
                "allow_aug": self.allow_complex_augmentation,
            })

        if no_valid_count > 0:
            print(f"Warning: {no_valid_count} images in {self.ann_file} had no valid annotations and were skipped.")

        return samples
