# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import math
import random
import copy
import PIL
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F



def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")


    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size: long side target length

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size

        # ---- resize long side ----
        if w > h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        # 可选：限制短边最小值或其他需求
        if max_size is not None:
            max_original_size = float(max((ow, oh)))
            if max_original_size > max_size:
                scale = max_size / max_original_size
                ow = int(round(ow * scale))
                oh = int(round(oh * scale))

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig)
        for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()

    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        masks = F.interpolate(
            target["masks"][:, None].float(),
            size=size,
            mode="nearest-exact"
        )[:, 0] > 0.5
        target["masks"] = masks.to(torch.bool)

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target

def pad_to_square(image, target, size, fill=0):
    # 严格限定只接收 torch.Tensor
    # assert isinstance(image, torch.Tensor), "输入图像必须是 torch.Tensor"
    
    # 假设 Tensor 格式为 [C, H, W]
    h, w = image.shape[1], image.shape[2]

    pad_h = size - h
    pad_w = size - w

    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    # F.pad 支持对 [C, H, W] 的 Tensor 进行边界填充
    image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

    if target is None:
        return image, None

    target = target.copy()

    # 针对归一化 cxcywh 坐标进行精确映射
    if "boxes" in target and target["boxes"].shape[0] > 0:
        boxes = target["boxes"].clone()
        
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # 换算公式说明：
        # 1. 还原像素坐标: cx * w
        # 2. 加上物理填充: + pad_left
        # 3. 按照新画布(size x size)重新归一化: / size
        boxes[:, 0] = (cx * w + pad_left) / size   # 新归一化 cx
        boxes[:, 1] = (cy * h + pad_top) / size    # 新归一化 cy
        boxes[:, 2] = (bw * w) / size              # 新归一化 w
        boxes[:, 3] = (bh * h) / size              # 新归一化 h
        
        target["boxes"] = boxes

    if "masks" in target and target["masks"] is not None:
        # torch.nn.functional.pad 的填充顺序是 (左, 右, 上, 下)
        target["masks"] = torch.nn.functional.pad(
            target["masks"],
            (pad_left, pad_right, pad_top, pad_bottom),
            value=0
        )

    target["size"] = torch.tensor([size, size], device=image.device)

    return image, target


class PadToSquare(object):
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img, target):
        return pad_to_square(img, target, self.size, self.fill)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p and target.get('allow_aug', False):
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

class ColorJitter(T.ColorJitter):
    """包装一下 torchvision 的 ColorJitter，让它支持 (img, target) 接口"""
    def __call__(self, img, target):
        if target is not None and not target.get('allow_aug', False):
            return img, target
        return super().__call__(img), target


class RandomGrayscale(T.RandomGrayscale):
    """包装 torchvision 的 RandomGrayscale，支持 (img, target) 接口"""
    def __call__(self, img, target):
        return super().__call__(img), target


class RandomGaussianBlur(object):
    """
    随机高斯模糊，支持 (img, target) 接口。
    以一定概率应用高斯模糊。
    """
    def __init__(self, p=0.5, kernel_size=(5, 9), sigma=(0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img, target):
        if random.random() < self.p and target.get('allow_aug', False):
            # 随机选择 kernel_size（奇数）
            if isinstance(self.kernel_size, tuple):
                min_k, max_k = self.kernel_size
                kernel_size = random.randrange(min_k, max_k + 1, 2)  # 确保奇数
            else:
                kernel_size = self.kernel_size
            
            # 随机选择 sigma
            if isinstance(self.sigma, tuple):
                sigma = random.uniform(self.sigma[0], self.sigma[1])
            else:
                sigma = self.sigma
            
            img = F.gaussian_blur(img, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
        
        return img, target

class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomTranslate(object):
    """
    随机平移图像和 bbox。
    平移后超出图像边界的 bbox 会被裁剪或过滤。
    """
    def __init__(self, p=0.5, translate=(0.1, 0.1), fill=114):
        self.p = p
        self.translate = translate  # (tx, ty) 最大平移比例
        self.fill = fill

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target
        
        w, h = img.size
        max_dx = int(self.translate[0] * w)
        max_dy = int(self.translate[1] * h)
        
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        
        # 平移图像
        img = F.affine(img, angle=0, translate=(dx, dy), scale=1.0, shear=0, fill=self.fill)
        
        if target is None:
            return img, target
        
        target = target.copy()
        
        if "boxes" in target:
            boxes = target["boxes"].clone()
            # 调整 bbox 坐标
            boxes[:, [0, 2]] += dx
            boxes[:, [1, 3]] += dy
            
            # 裁剪到图像边界内
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
            
            # 计算有效框（宽度和高度都大于0）
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            
            target["boxes"] = boxes[keep]
            
            # 同步过滤其他字段
            for field in ["labels", "area"]:
                if field in target:
                    target[field] = target[field][keep]
            
            if "masks" in target:
                target["masks"] = target["masks"][keep]
        
        return img, target


class RandomScale(object):
    """
    随机缩放图像和 bbox。
    缩放后保持图像大小不变（通过填充或裁剪）。
    """
    def __init__(self, p=0.5, scale=(0.8, 1.2), fill=114):
        self.p = p
        self.scale = scale  # (min_scale, max_scale)
        self.fill = fill

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target
        
        scale = random.uniform(self.scale[0], self.scale[1])
        w, h = img.size
        
        # 计算新尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        img = F.resize(img, (new_h, new_w))
        
        if target is not None:
            target = target.copy()
            
            if "boxes" in target:
                boxes = target["boxes"].clone()
                boxes = boxes * scale
                target["boxes"] = boxes
            
            if "area" in target:
                target["area"] = target["area"] * (scale ** 2)
            
            if "masks" in target:
                masks = target["masks"].float()
                masks = F.resize(masks, (new_h, new_w), interpolation=F.InterpolationMode.NEAREST)
                target["masks"] = masks > 0.5
        
        # 如果缩放后尺寸不同，需要 pad 或 crop 回原尺寸
        if new_w != w or new_h != h:
            if scale > 1.0:
                # 放大后需要裁剪到原尺寸（中心裁剪）
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                img = F.crop(img, top, left, h, w)
                
                if target is not None and "boxes" in target:
                    boxes = target["boxes"].clone()
                    boxes[:, [0, 2]] -= left
                    boxes[:, [1, 3]] -= top
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
                    
                    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    target["boxes"] = boxes[keep]
                    
                    for field in ["labels", "area"]:
                        if field in target:
                            target[field] = target[field][keep]
                    
                    if "masks" in target:
                        target["masks"] = target["masks"][keep]
                        target["masks"] = F.crop(target["masks"], top, left, h, w)
            else:
                # 缩小后需要填充回原尺寸
                pad_w = w - new_w
                pad_h = h - new_h
                pad_left = pad_w // 2
                pad_top = pad_h // 2
                pad_right = pad_w - pad_left
                pad_bottom = pad_h - pad_top
                
                img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
                
                if target is not None and "boxes" in target:
                    boxes = target["boxes"].clone()
                    boxes[:, [0, 2]] += pad_left
                    boxes[:, [1, 3]] += pad_top
                    target["boxes"] = boxes
                    
                    if "masks" in target:
                        target["masks"] = F.pad(
                            target["masks"],
                            (pad_left, pad_right, pad_top, pad_bottom)
                        )
        
        return img, target


class RandomErasing(object):
    """
    随机擦除图像区域（用于 Tensor 输入）。
    如果擦除区域与 bbox 重叠过多，会过滤该 bbox。
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, max_overlap_ratio=0.5):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.max_overlap_ratio = max_overlap_ratio  # 允许的最大重叠比例

    def __call__(self, image, target):
        """
        Args:
            image: Tensor of shape (C, H, W)
            target: dict with "boxes" (xyxy format)
        """
        if random.random() > self.p:
            return image, target
        
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"RandomErasing expects torch.Tensor, got {type(image)}")
        
        C, H, W = image.shape
        
        # 随机生成擦除区域
        area = H * W
        
        # ====================== 生成擦除区域（稳定版）======================
        erased = False
        for _ in range(10):                     # 最多尝试10次
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            h = int(math.sqrt(target_area / aspect_ratio))
            w = int(target_area / h) if h > 0 else 0

            if 1 <= h < H and 1 <= w < W:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                erased = True
                break

        if not erased:
            return image, target   # 实在生成不了就跳过
        
        # 执行擦除
        if isinstance(self.value, (int, float)):
            image[:, top:top + h, left:left + w] = self.value
        elif isinstance(self.value, str) and self.value.lower() == "random":
            noise = torch.randn(C, h, w, device=image.device, dtype=image.dtype) * 0.5
            image[:, top:top + h, left:left + w] = noise.clamp(0, 1) if image.dtype == torch.float32 else noise
        
        if "boxes" not in target or len(target["boxes"]) == 0:
            return image, target
        
        # 深拷贝避免修改原 dict
        target = copy.deepcopy(target)
        boxes = target["boxes"]          # [N, 4] cxcywh normalized [0,1]

        # cxcywh normalized → xyxy pixel（核心转换）
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H

        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)   # [N, 4] pixel

        # 擦除区域的 xyxy
        erase_box = torch.tensor([left, top, left + w, top + h],
                                 dtype=boxes_xyxy.dtype,
                                 device=boxes_xyxy.device)

        # 计算重叠面积
        inter_x1 = torch.maximum(boxes_xyxy[:, 0], erase_box[0])
        inter_y1 = torch.maximum(boxes_xyxy[:, 1], erase_box[1])
        inter_x2 = torch.minimum(boxes_xyxy[:, 2], erase_box[2])
        inter_y2 = torch.minimum(boxes_xyxy[:, 3], erase_box[3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        box_area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

        overlap_ratio = inter_area / (box_area + 1e-6)

        # 保留重叠比例小于阈值的实例
        keep = overlap_ratio < self.max_overlap_ratio

        # 过滤所有相关字段
        target["boxes"] = boxes[keep]

        for field in ["labels"]:
            if field in target:
                target[field] = target[field][keep]

        # ====================== 处理 mask（分割专用）======================
        # TODO: implement mask erasing if needed (较复杂)
        #if "masks" in target:                                   # binary masks [N, H, W]
        #    target["masks"] = target["masks"][keep]
        #    if self.erase_mask_pixels:
        #        # 对保留的 mask 也擦除对应区域（避免 label noise）
        #        target["masks"][:, top:top + h, left:left + w] = 0

        #elif "segmentations" in target or "segments" in target:
            # polygon 格式通常只过滤整个实例（像素级擦除较复杂）
            #key = "segmentations" if "segmentations" in target else "segments"
            #target[key] = [target[key][i] for i in range(len(target[key])) if keep[i]]

        return image, target


class Mosaic(object):
    def __init__(self, p=0.5, img_size=640, max_cache_size=50):
        """
        Args:
            p (float): 触发 Mosaic 增强的概率
            img_size (int): 最终输出的图像尺寸 (H, W)
            max_cache_size (int): 内部缓存的最大样本数
        """
        self.p = p
        self.img_size = img_size
        self.max_cache_size = max_cache_size
        self.cache = []

    def _update_cache(self, img, target):
        if target.get('allow_aug', False):
            if len(self.cache) >= self.max_cache_size:
                self.cache.pop(random.randint(0, len(self.cache) - 1))
            
            cached_img = img.clone() if isinstance(img, torch.Tensor) else img.copy()
            cached_target = {
                "image_id": target["image_id"].clone(),
                "boxes": target["boxes"].clone(), # 内部继续保留原本的 cxcywh 归一化
                "labels": target["labels"].clone(),
                "captions": copy.deepcopy(target["captions"]),
                # "segmentations": copy.deepcopy(target["segmentations"]),
                "text_is_positive": copy.deepcopy(target["text_is_positive"]),
                "allow_aug": target["allow_aug"]
            }
            self.cache.append({'img': cached_img, 'target': cached_target})

    def __call__(self, img, target):       
        
        if (random.random() < self.p) and target.get('allow_aug', False) and (len(self.cache) >= 3):
            mimg, mtarget = self._apply_mosaic(img, target)
            self._update_cache(img, target) 
            return mimg, mtarget
        if target.get('allow_aug', False):
            self._update_cache(img, target) 
        return img, target

    def _apply_mosaic(self, current_img, current_target):
        mix_samples = random.sample(self.cache, 3)
        all_samples = [{'img': current_img, 'target': current_target}] + mix_samples
        random.shuffle(all_samples)

        s = self.img_size
        # 随机选择大图(2S x 2S)内部的十字拼接中心点 (xc, yc)
        # 为了保证四张图都能露脸，中心点一般限制在 s//2 到 3s//2 之间
        xc = int(random.uniform(s // 2, 3 * s // 2))
        yc = int(random.uniform(s // 2, 3 * s // 2))

        # 2倍大小的大画布 [C, 2S, 2S]
        mosaic_img = torch.full((current_img.shape[0], s * 2, s * 2), 0.447, dtype=current_img.dtype)

        # -------------------------------------------------------------
        # 【保持之前完美的文本重组逻辑】
        target_len = len(current_target['captions'])
        all_pos_caps, all_neg_caps = set(), set()
        for sample in all_samples:
            for cap, pos_flag in zip(sample['target']['captions'], sample['target']['text_is_positive']):
                if pos_flag: all_pos_caps.add(cap)
                else: all_neg_caps.add(cap)
        list_pos_caps = list(all_pos_caps)
        list_neg_caps = list(all_neg_caps - all_pos_caps)
        selected_pos = []
        selected_neg = []
        if len(list_pos_caps) >= target_len:
            selected_pos = random.sample(list_pos_caps, target_len)
        else:
            selected_pos = list_pos_caps
            needed_neg_len = target_len - len(selected_pos)
            selected_neg = random.sample(list_neg_caps, needed_neg_len) if len(list_neg_caps) >= needed_neg_len else list_neg_caps
            while len(selected_pos) + len(selected_neg) < target_len:
                selected_neg.append(random.choice(list_neg_caps) if len(list_neg_caps) > 0 else "background_pad")

        global_captions = selected_pos + selected_neg
        global_text_is_positive = [True] * len(selected_pos) + [False] * len(selected_neg)
        caption_to_global_idx = {cap: idx for idx, cap in enumerate(global_captions)}
        # -------------------------------------------------------------

        mosaic_boxes = []
        mosaic_labels = []
        # mosaic_segs = []

        for i, sample in enumerate(all_samples):
            simg = sample['img']
            starget = sample['target']
            cxcywh_boxes = starget["boxes"]
            labels = starget["labels"].flatten()
            captions = starget["captions"]
            # segs = starget["segmentations"]

            h0, w0 = simg.shape[1], simg.shape[2] # 原始宽高

            # --- 【全新升级 1】将子图缩放到目标尺寸 s, 计算缩放比例 ---
            # 这样保证子图能充满它所在的象限区域，杜绝留白
            r = s / max(h0, w0)
            if r != 1: 
                simg = F.resize(simg, [int(h0 * r), int(w0 * r)])
            h, w = simg.shape[1], simg.shape[2] # 缩放后的宽高

            # 转换当前图的 bbox 标签到全局文本池（保留未被截断的框）
            valid_box_indices = []
            remapped_labels_list = []
            for box_idx, local_label in enumerate(labels):
                cap_text = captions[int(local_label)]
                if cap_text in caption_to_global_idx:
                    valid_box_indices.append(box_idx)
                    remapped_labels_list.append(caption_to_global_idx[cap_text])

            # 将归一化 cxcywh 直接转换为【基于缩放后图像】的绝对像素坐标 xyxy
            abs_boxes = torch.zeros_like(cxcywh_boxes)
            if cxcywh_boxes.shape[0] > 0:
                cx, cy, bw, bh = cxcywh_boxes[:, 0], cxcywh_boxes[:, 1], cxcywh_boxes[:, 2], cxcywh_boxes[:, 3]
                abs_boxes[:, 0] = (cx - bw / 2) * w
                abs_boxes[:, 1] = (cy - bh / 2) * h
                abs_boxes[:, 2] = (cx + bw / 2) * w
                abs_boxes[:, 3] = (cy + bh / 2) * h

            if len(valid_box_indices) == 0:
                abs_boxes = torch.zeros((0, 4), dtype=torch.float32)
                remapped_labels = torch.zeros((0,), dtype=torch.int64)
            else:
                abs_boxes = abs_boxes[valid_box_indices]
                remapped_labels = torch.tensor(remapped_labels_list, dtype=torch.int64, device=labels.device)
                # segs = [segs[idx] for idx in valid_box_indices]

            # --- 【全新升级 2】行业标准的无缝四象限绝对坐标计算 ---
            # 无论原图多大，强行贴满边界并汇聚于中心点 (xc, yc)
            if i == 0:    # 左上象限
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # 右上象限
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # 左下象限
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 3:  # 右下象限
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            # 无缝拼入大图
            mosaic_img[:, y1a:y2a, x1a:x2a] = simg[:, y1b:y2b, x1b:x2b]

            # 计算坐标平移量（大图起始坐标 - 缩放后子图的截取起始坐标）
            padw = x1a - x1b
            padh = y1a - y1b

            if abs_boxes.shape[0] > 0:
                img_boxes = abs_boxes.clone()
                img_boxes[:, [0, 2]] += padw
                img_boxes[:, [1, 3]] += padh
                
                # 精确截断超出各自象限贴图边界的 boxes
                img_boxes[:, 0] = img_boxes[:, 0].clamp(min=x1a, max=x2a)
                img_boxes[:, 2] = img_boxes[:, 2].clamp(min=x1a, max=x2a)
                img_boxes[:, 1] = img_boxes[:, 1].clamp(min=y1a, max=y2a)
                img_boxes[:, 3] = img_boxes[:, 3].clamp(min=y1a, max=y2a)

                w_box = img_boxes[:, 2] - img_boxes[:, 0]
                h_box = img_boxes[:, 3] - img_boxes[:, 1]
                # 过滤掉由于拼接裁剪导致完全被切到画面外的无效框
                keep = (w_box > 1) & (h_box > 1)

                if keep.any():
                    mosaic_boxes.append(img_boxes[keep])
                    mosaic_labels.append(remapped_labels[keep])
                    
                    keep_cpu = keep.cpu().numpy()
                    # for idx, is_keep in enumerate(keep_cpu):
                    #     if is_keep:
                    #         mosaic_segs.append(segs[idx])

        # --- 步骤 5: 汇总并将大图归一化转回 cxcywh ---
        out_target = copy.deepcopy(current_target)
        
        if len(mosaic_boxes) > 0:
            out_boxes_xyxy = torch.cat(mosaic_boxes, dim=0)
            out_labels = torch.cat(mosaic_labels, dim=0)
            
            # 重新归一化到大图总宽高 (2 * s)
            out_boxes_xyxy /= (2 * s)
            
            out_boxes_cxcywh = torch.zeros_like(out_boxes_xyxy)
            out_boxes_cxcywh[:, 0] = (out_boxes_xyxy[:, 0] + out_boxes_xyxy[:, 2]) / 2
            out_boxes_cxcywh[:, 1] = (out_boxes_xyxy[:, 1] + out_boxes_xyxy[:, 3]) / 2
            out_boxes_cxcywh[:, 2] = out_boxes_xyxy[:, 2] - out_boxes_xyxy[:, 0]
            out_boxes_cxcywh[:, 3] = out_boxes_xyxy[:, 3] - out_boxes_xyxy[:, 1]
            
            if len(current_target["labels"].shape) == 2 and current_target["labels"].shape[1] == 1:
                out_labels = out_labels.unsqueeze(-1)
        else:
            out_boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32)
            out_labels = torch.zeros((0, 1) if len(current_target["labels"].shape) == 2 else (0,), dtype=torch.int64)

        # 缩放大图 2S -> S 输出
        final_img = F.resize(mosaic_img, [s, s])

        out_target["boxes"] = out_boxes_cxcywh
        out_target["labels"] = out_labels
        out_target["captions"] = global_captions
        # out_target["segmentations"] = mosaic_segs
        out_target["text_is_positive"] = torch.tensor(global_text_is_positive, dtype=torch.bool)

        return final_img, out_target


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    
def make_coco_transforms(istrain, target_size=1024):

    normalize = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    # ConvNeXt 常用短边范围（可根据你的显存/模型大小调整）
    # 推荐范围：短边从 ~0.7×target 到 ~1.25×target
    short_edge_scales = [int(target_size * r) for r in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25]]

    # 或者更宽松一些（类似 DINO / RT-DETR 的高分辨率训练风格）
    # short_edge_scales = list(range(512, 1408, 32))   # 步长32，兼容很多 stride=32 的 backbone

    if istrain:
        return Compose([
            RandomHorizontalFlip(p=0.5),
            # 几何变换
            RandomTranslate(p=0.3, translate=(0.1, 0.1), fill=114),
            RandomScale(p=0.3, scale=(0.8, 1.2), fill=114),
            # 色彩 + 模糊组合（强度中等）
            ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.15),
            #RandomGrayscale(p=0.08),
            RandomGaussianBlur(p=0.3, kernel_size=(3, 7), sigma=(0.1, 1.5)),
            Resize(target_size), #必须先resize长边
            normalize,
            Mosaic(0.8, target_size), #放在ToTensor后获得更佳性能
            PadToSquare(target_size, fill=0.47),   # 最终 pad 到 target_size × target_size
            # RandomErasing 必须在 ToTensor 之后，对 Tensor 操作
            # RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0, max_overlap_ratio=0.5),
        ])

    else:  # val / test
        return Compose([
            Resize(target_size),              # 长边 resize 到 target_size，保持比例
            normalize,
            PadToSquare(target_size, fill=0.47),
        ])