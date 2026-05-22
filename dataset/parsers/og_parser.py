# dataset/parsers/og_parser.py
import json
import os
from collections import defaultdict
from typing import List, Dict, Any

from .base import BaseParser


class OGParser(BaseParser):
    def parse(self) -> List[Dict[str, Any]]:
        with open(self.ann_file, "r") as f:
            data = json.load(f)

        images = {img["id"]: img for img in data["images"]}

        ann_map = defaultdict(list)
        for ann in data["annotations"]:
            ann_map[ann["image_id"]].append(ann)

        samples = []

        for image_id, img in images.items():
            anns = ann_map.get(image_id, [])

            boxes = []
            labels = []
            captions = []
            segmentations = []

            for ann in anns:
                boxes.append(ann["bbox"])                  # xywh
                labels.append(ann["category_id"])          # usually -1
                captions.append(ann.get("caption", ""))    # open-vocab
                segmentations.append(ann.get("segmentation", []))

            samples.append({
                "image_id": image_id,
                "file_name": os.path.join(self.image_root, img["file_name"]),
                "height": img["height"],
                "width": img["width"],

                "boxes": boxes,
                "labels": labels,
                "captions": captions,
                # "segmentations": segmentations,
            })

        return samples
