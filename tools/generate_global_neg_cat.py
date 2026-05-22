import numpy as np
from pathlib import Path
from collections import defaultdict
from collections import Counter
import os
import json
from tqdm import tqdm

def get_caption_counts(json_files_dict, min_freq=100):
    caption_list = []
    for path in json_files_dict.values():
        """提取单个 JSON 文件中的 Caption 频次并排序"""
        print(f"\n正在处理: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_map = {img['id']: img for img in data.get('images', [])}

        for ann in tqdm(data.get('annotations', []), desc="解析标注"):
            caption = None
            # 兼容模式 1: 直接包含 caption
            if 'caption' in ann:
                caption = str(ann['caption']).lower().strip()
            # 兼容模式 2: tokens_positive 模式
            elif "tokens_positive" in ann and ann["tokens_positive"]:
                img = img_map.get(ann.get('image_id'))
                if img and "caption" in img:
                    try:
                        phrase_parts = [img["caption"][t[0]:t[1]] for t in ann["tokens_positive"]]
                        caption = " ".join(phrase_parts).lower().strip()
                    except: continue
            
            if caption:
                caption_list.append(caption)

    # 统计与降序排列
    counts_dict = Counter(caption_list)
    # 筛选出现频率大于阈值的 caption
    high_freq_captions = [
        (caption, count) 
        for caption, count in counts_dict.items() 
        if count > min_freq
    ]
    
    # 按频次降序排列
    high_freq_captions.sort(key=lambda x: x[1], reverse=True)
    
    # 显式清理大对象内存
    del data
    del img_map
    del caption_list
    
    return high_freq_captions, counts_dict


# --- 使用示例 ---
if __name__ == "__main__":
    # 在这里配置你的数据集名称和对应的路径
    datasets = {
        "DeepFashion2": "/root/autodl-tmp/OOD/DeepFashion2/annotations/deepfasion2_with_caption.json",
        "Objects365_caption": "/root/autodl-tmp/OOD/Objects365_v1/annotations/objects365_train_with_caption.json",
        "reircoco": "/root/autodl-tmp/OOD/refcoco/annotations/reircoco_train_segm.json",
        "MixGrounding": "/root/autodl-tmp/OOD/MixedGrounding/mdetr_annotations/final_mixed_train_no_coco_segm_fixed.json",
        "grefcoco": "/root/autodl-tmp/OOD/refcoco/annotations/grefcoco_sgem.json",
        "Flickr30k": "/root/autodl-tmp/OOD/flickr30k/final_flickr_separateGT_train_segm.json",
        "refcoco+(train)": "/root/autodl-tmp/OOD/refcoco/annotations/refcoco+_unc_train_coco_segm.json",
        "refcocog(train)": "/root/autodl-tmp/OOD/refcoco/annotations/refcocog_umd_train_coco_segm.json",
        "refcoco(train)": "/root/autodl-tmp/OOD/refcoco/annotations/refcoco_unc_train_coco_segm.json",
    }

    min_frequency = 100
    
    high_freq_list, all_counts = get_caption_counts(datasets, min_freq=min_frequency)
    
    print(f"\n{'='*50}")
    print(f"出现频率 > {min_frequency} 的 Caption 共 {len(high_freq_list)} 个")
    print(f"{'='*50}")
    
    # 打印前20个高频 caption
    for i, (caption, count) in enumerate(high_freq_list[:20], 1):
        print(f"{i:3d}. [{count:4d}次]  {caption}")
    
    # 如果需要纯 caption 列表（不含频次）
    caption_only_list = [caption for caption, count in high_freq_list]
    print(f"\n纯 caption 列表长度: {len(caption_only_list)}")

    # 保存为 JSON 文件
    output_path = "/root/autodl-tmp/VLMs/dataset/global_grounding_neg_cat.json"  # 修改为你想要的路径
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(caption_only_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存 {len(caption_only_list)} 个高频 caption 到: {output_path}")
    