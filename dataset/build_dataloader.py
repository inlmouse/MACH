# dataset/build_dataloader.py
import json
import os
from collections import Counter
from typing import List, Dict, Any, Optional
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

from .parsers.og_parser import OGParser
from .parsers.coco_parser import COCOParser
from .unified_dataset import UnifiedDetectionDataset

from .transforms import make_coco_transforms


def load_labels(labels_file: str = "labels.txt") -> Optional[List[str]]:
    """
    从 txt 文件加载 label 映射
    
    Args:
        labels_file: label 文件路径
    
    Returns:
        List[str]: 类别名称列表，索引即为 label id
        None: 文件不存在
    
    Example:
        >>> labels = load_labels("labels.txt")
        >>> print(labels[0])  # 输出 label 0 对应的类别名称
    """
    if not os.path.exists(labels_file):
        print(f"Warning: Label file {labels_file} not found")
        return None
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(labels)} classes from {labels_file}")
    return labels


def detect_format(ann_file: str) -> str:
    return "coco"
    with open(ann_file, "r") as f:
        data = json.load(f)

    if "images" in data and "annotations" in data:
        # OG 特征：annotation 里有 caption
        if len(data["annotations"]) > 0 and "caption" in data["annotations"][0]:
            return "og"
        return "coco"

    raise ValueError(f"Unknown annotation format: {ann_file}")

def remap_all_labelcaptions(data: List[Dict[str, Any]], save_path: str = None) -> int:
    """
    把所有 labels == -2 的 captions 收集起来，
    统计频率 → 按频率降序分配新 label (0 开始)
    然后把这些新 label 写回到原来是 -2 的位置
    
    同时保存 label → caption 映射到 txt 文件，每行一个 caption，按 label id 排序
    
    注意：会直接修改传入的 data（in-place）
    
    Returns:
        num_classes: 类别数
    """
    # 第一步：收集所有 -2 对应的 caption
    caption_count = Counter()
    
    for item in tqdm(data, desc="select all labeled captions", unit="item"):
        labels = item.get("labels", [])
        captions = item.get("captions", [])
        
        if len(labels) != len(captions):
            continue  # 数据不合法，跳过（或者你可以 raise）
            
        for label, cap in zip(labels, captions):
            if label == -2:
                caption_count[cap] += 1
    
    if not caption_count:
        return []  # 没有任何 -2，返回空列表
    
    # 第二步：按出现次数降序排序，频率相同则保持首次出现的相对顺序（stable）
    # 最常见 → 排在最前面 → 得到 label 0
    sorted_captions = [cap for cap, _ in caption_count.most_common()]
    num_classes = len(sorted_captions)

    print(f"发现 {num_classes} 个不同的未知 caption（-2 对应的）")
    print("按出现频率排序的前 5 个（如果有）：")
    for i, cap in enumerate(sorted_captions[:5], 1):
        print(f"  {i}. {cap} ({caption_count[cap]} 次)")
    if num_classes > 5:
        print(f"  ... 共 {num_classes} 类")
    
    # 第三步：创建 caption → 新 label 的映射
    caption_to_new_label = {
        cap: new_id
        for new_id, cap in enumerate(sorted_captions)
    }
    
    # 第四步：把 -2 的位置替换成对应的新 label
    for item in tqdm(data, desc="re-map all labels", unit="item"):
        labels = item["labels"]
        captions = item["captions"]
        
        for i in range(len(labels)):
            if labels[i] == -2:
                cap = captions[i]
                labels[i] = caption_to_new_label[cap]
    
    # 第五步：保存 label → caption 映射到 txt 文件
    # sorted_captions[0] 对应 label 0，sorted_captions[1] 对应 label 1，以此类推
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            for i, caption in enumerate(sorted_captions):
                f.write(f"{caption}\n")
        print(f"Label 映射已保存到: {save_path}")
    
    return sorted_captions


def load_annotations(ann_files, image_roots, allow_complex_augmentation=None):
    """
    ann_files: List[str]
    image_roots: List[str]
    """
    assert len(ann_files) == len(image_roots)
    if allow_complex_augmentation is not None:
        assert len(ann_files) == len(allow_complex_augmentation)
    else:
        allow_complex_augmentation = [False] * len(ann_files)
    all_samples = []

    for ann_file, img_root, allow_aug in zip(ann_files, image_roots, allow_complex_augmentation):
        fmt = detect_format(ann_file)

        if fmt == "og":
            parser = OGParser(ann_file, img_root)
        elif fmt == "coco":
            parser = COCOParser(ann_file, img_root, allow_aug)
        else:
            raise ValueError(fmt)

        all_samples.extend(parser.parse())

    return all_samples

def debug_label_coverage(b_idx, l_idx, num_classes=None):
    b_idx = b_idx.long()
    l_idx = l_idx.view(-1).long()

    B = int(b_idx.max().item()) + 1

    print("=" * 50)
    print("[Check label continuity per batch]\n")

    for b in range(B):
        mask = (b_idx == b)
        labels = l_idx[mask]

        if labels.numel() == 0:
            print(f"Batch {b}: ⚠️ EMPTY")
            continue

        uniq = torch.unique(labels)
        uniq_sorted = torch.sort(uniq).values

        # 构造理想连续序列
        expected = torch.arange(len(uniq_sorted), device=uniq.device)

        is_contiguous = torch.equal(uniq_sorted, expected)

        print(f"Batch {b}: labels = {uniq_sorted.tolist()}")

        if not is_contiguous:
            missing = set(range(int(uniq_sorted.max().item()) + 1)) - set(uniq_sorted.tolist())
            print(f"  ❌ NOT CONTIGUOUS")
            print(f"  missing in [0, max]: {sorted(list(missing))}")
        #else:
        #    print(f"  ✅ contiguous from 0")

def build_dataset(
    ann_files,
    image_roots,
    allow_complex_augmentation=None,
    num_infonce_batch=None,
    istrain=True,
    target_size=640,
    labels_file: str = None,
):
    """
    构建数据集，返回 (dataset, num_classes)
    
    Args:
        ann_files: 标注文件路径列表
        image_roots: 图片根目录列表
        istrain: 是否训练模式
        target_size: 目标尺寸
        labels_file: 保存 label 映射的文件路径（txt格式，每行一个类别名称）
    """
    samples = load_annotations(ann_files, image_roots, allow_complex_augmentation)
    label_list = remap_all_labelcaptions(samples, save_path=labels_file)
        
    dataset = UnifiedDetectionDataset(
        samples=samples,
        label_list=label_list,
        num_infonce_batch=num_infonce_batch,
        transforms=make_coco_transforms(istrain, target_size=target_size),
        istrain=istrain,
    )
    return dataset


def build_dataloader(
    ann_files=None,
    image_roots=None,
    allow_complex_augmentation=None,
    dataset=None,
    num_infonce_batch=None,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    istrain=True,
    target_size=640,
    sampler=None,
    pin_memory=False,
    labels_file: str = "labels.txt",
):
    """
    构建 DataLoader。
    两种方式:
    1. 传入 ann_files 和 image_roots 自动构建 dataset
    2. 直接传入已构建好的 dataset（此时不需要 ann_files/image_roots）
    
    Args:
        ann_files: 标注文件路径列表（方式1需要）
        image_roots: 图片根目录列表（方式1需要）
        dataset: 已构建好的 UnifiedDetectionDataset（方式2需要）
        labels_file: label 映射保存路径（仅在方式1构建 dataset 时使用）
    """
    
    if dataset is None:
        # 方式1：从 ann_files 和 image_roots 构建 dataset
        assert ann_files is not None and image_roots is not None, \
            "当 dataset 为 None 时，必须提供 ann_files 和 image_roots"
        dataset = build_dataset(ann_files, image_roots, allow_complex_augmentation, num_infonce_batch, istrain, target_size, labels_file)
    else:
        # 方式2：使用传入的 dataset，忽略 ann_files/image_roots
        pass


    def collate_fn(batch: List[tuple[torch.Tensor, Dict[str, Any]]]):
        """
        batch: List[(image: Tensor[C,H,W], target: Dict)]
        
        返回:
        {
            'img':              Tensor[B, C, H, W]              # 堆叠后的图像
            'targets':          List[Dict]                      # 完整原始 target 列表，长度 B
            'batch_idx':        Tensor[总 gt 数]                # 每个 gt 属于哪张图 (0 ~ B-1)
            'cls':              Tensor[总 gt 数, ] 或 [总 gt 数, 1]
            'bboxes':           Tensor[总 gt 数, 4]
        }
        """
        images = []
        textfeats = []
        batch_captions = []
        targets_list = []           # 保留完整 target dict 列表
        batch_idx = []
        cls_all = []
        bboxes_all = []
        text_is_positive = []

        for img_idx, (img, target) in enumerate(batch):
            images.append(img)
            targets_list.append(target)  # 完整保留，不动它

            # 提取 labels 和 boxes
            labels = target.get("labels", torch.empty(0, dtype=torch.int64))
            boxes = target.get("boxes", torch.empty((0, 4), dtype=torch.float32))

            n = len(labels) if labels.numel() > 0 else 0

            if n > 0:
                batch_idx.extend([img_idx] * n)
                
                # labels 通常是 [n]，loss 常用 [n,1] 或 [n]
                # 这里统一转成 [n,1]
                if labels.ndim == 1:
                    labels = labels.unsqueeze(1)
                cls_all.append(labels.float())   # BCEWithLogitsLoss 期望 float
                
                bboxes_all.append(boxes)

            if target.get("captions") is not None:
                batch_captions.extend(target["captions"])  # List[num_infonce_batch] of str

            if target.get("text_is_positive") is not None:
                text_is_positive.append(target["text_is_positive"])  # [B, num_infonce_batch] bool

        # 转 tensor
        images = torch.stack(images) if images else torch.empty((0, 3, 640, 640))
        
        text_is_positive = torch.stack(text_is_positive, dim=0).bool() if text_is_positive else None
        batch_idx = torch.tensor(batch_idx, dtype=torch.long) if batch_idx else torch.empty((0,), dtype=torch.long)
        gt_groups = [(batch_idx == i).sum().item() for i in range(images.shape[0])]
        cls = torch.cat(cls_all, dim=0) if cls_all else torch.empty((0, 1), dtype=torch.float32)
        bboxes = torch.cat(bboxes_all, dim=0) if bboxes_all else torch.empty((0, 4), dtype=torch.float32)

        #debug_label_coverage(batch_idx, cls, None)

        return {
                'img': images,
                'batch_captions': batch_captions,           # 文本特征，形状 List[B*num_infonce_batch] of str
                'text_is_positive': text_is_positive,  # 文本正负样本标签，形状 [B, num_infonce_batch] bool，如果没有则是 [0] bool
                'targets': targets_list,          # 完整原始 dict 列表，可供后续使用（如可视化、计算 mAP 等）
                'batch_idx': batch_idx,
                'gt_groups': gt_groups,
                'cls': cls,
                'bboxes': bboxes,
            }
            
        

    # 如果传入了 sampler，则禁用 shuffle（由 sampler 控制）
    effective_shuffle = shuffle if sampler is None else False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )