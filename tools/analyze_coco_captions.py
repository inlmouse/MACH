import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm

def get_caption_counts(json_path):
    """提取单个 JSON 文件中的 Caption 频次并排序"""
    print(f"\n正在处理: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_map = {img['id']: img for img in data.get('images', [])}
    caption_list = []

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
    sorted_counts = sorted(counts_dict.values(), reverse=True)
    
    # 显式清理大对象内存
    del data
    del img_map
    del caption_list
    
    return sorted_counts

def plot_multiple_distributions(json_files_dict):
    """
    json_files_dict: 字典格式 { "数据集名称": "路径.json" }
    """
    plt.figure(figsize=(8, 6))
    
    # 使用较美观的配色方案
    colors = plt.cm.tab10(np.linspace(0, 1, len(json_files_dict)))
    
    for (label, path), color in zip(json_files_dict.items(), colors):
        sorted_counts = get_caption_counts(path)
        
        if not sorted_counts:
            print(f"警告: {label} 未提取到有效 Caption")
            continue
            
        x = np.arange(len(sorted_counts))
        y = np.array(sorted_counts)
        
        # 绘图
        plt.plot(x, y, label=f"{label} (Unique: {len(sorted_counts)})", 
                 color=color, linewidth=2, alpha=0.8)

    # 设置 y 轴为对数
    #plt.xscale('log')
    plt.yscale('log')
    
    # 图表修饰
    plt.title('Caption Frequency Distribution Comparison', fontsize=16)
    plt.xlabel('Caption Rank (ID sorted by frequency)', fontsize=14)
    plt.ylabel('Frequency (Log Scale)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=12)
    
    # 自动调整布局，防止标签溢出
    plt.tight_layout()
    
    save_path = 'multi_dataset_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[完成] 对比图表已保存至: {save_path}")
    #plt.show()

def get_sorted_lengths_with_weights(json_path):
    """提取 Caption 及其频次，返回排序后的长度列表和加权平均长度"""
    print(f"\n正在加载数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_map = {img['id']: img for img in data.get('images', [])}
    
    # 1. 使用 Counter 统计所有 Caption 的出现次数
    caption_counter = Counter()

    for ann in tqdm(data.get('annotations', []), desc="解析"):
        caption = None
        if 'caption' in ann:
            caption = str(ann['caption']).lower().strip()
        elif "tokens_positive" in ann and ann["tokens_positive"]:
            img = img_map.get(ann.get('image_id'))
            if img and "caption" in img:
                try:
                    phrase_parts = [img["caption"][t[0]:t[1]] for t in ann["tokens_positive"]]
                    caption = " ".join(phrase_parts).lower().strip()
                except: continue
        
        if caption:
            caption_counter[caption] += 1

    # 2. 计算加权平均长度
    total_words = 0
    total_instances = 0
    
    # 存储唯一 Caption 的长度，用于后续绘图
    unique_lengths = []
    
    for caption, count in caption_counter.items():
        word_count = len(caption.split())
        unique_lengths.append(word_count)
        
        # 加权累计：长度 * 出现次数
        total_words += word_count * count
        total_instances += count

    weighted_avg = total_words / total_instances if total_instances > 0 else 0

    # 3. 将唯一 Caption 的长度降序排列（用于横坐标）
    unique_lengths.sort(reverse=True)
    
    del data
    del img_map
    del caption_counter
    
    return unique_lengths, weighted_avg, total_instances

def plot_weighted_length_comparison(json_files_dict):
    """
    绘制长度分布对比图
    横坐标：按长度降序排列的唯一 Caption 序号
    纵坐标：单词个数
    图例：展示加权平均长度
    """
    plt.figure(figsize=(8, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.7, len(json_files_dict)))

    for (label, path), color in zip(json_files_dict.items(), colors):
        unique_lengths, weighted_avg, total_instances = get_sorted_lengths_with_weights(path)
        
        if not unique_lengths:
            continue

        x = np.arange(len(unique_lengths))
        y = np.array(unique_lengths)
        max_len = np.max(y)
        
        # 绘图
        # 图例中展示真实的加权平均长度和总标注数
        label_text = f"{label} (W-Avg: {weighted_avg:.1f}, Max: {max_len}, Total Ann: {total_instances})"
        plt.plot(x, y, label=label_text, color=color, linewidth=2.5, alpha=0.9)

    plt.title('Caption Length Distribution Comparison', fontsize=16)
    plt.xlabel('Caption Rank (Unique Captions, Longest to Shortest)', fontsize=14)
    plt.ylabel('Number of Words', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    
    save_path = 'caption_weighted_length_rank.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[完成] 加权长度排序图已保存至: {save_path}")
    plt.show()


# --- 使用示例 ---
if __name__ == "__main__":
    # 在这里配置你的数据集名称和对应的路径
    datasets = {
        "Objects365_caption": "/data/OOD/Objects365_v1/annotations/objects365_train_with_caption.json",
        "reircoco": "/data/OOD/refer/data/annotations/reircoco_train_segm.json",
        "MixGrounding": "/data/OOD/MixedGrounding/mdetr_annotations/final_mixed_train_no_coco_segm.json",
        "grefcoco": "/data/OOD/refer/data/annotations/grefcoco_sgem.json",
        "Flickr30k": "/data/OOD/flickr30k/final_flickr_separateGT_train_segm.json",
        "refcoco+(train)": "/data/OOD/refer/data/annotations/refcoco+_unc_train_coco_segm.json",
        "refcocog(train)": "/data/OOD/refer/data/annotations/refcocog_umd_train_coco_segm.json",
        "refcoco(train)": "/data/OOD/refer/data/annotations/refcoco_unc_train_coco_segm.json",
    }
    
    plot_multiple_distributions(datasets)
    plot_weighted_length_comparison(datasets)
