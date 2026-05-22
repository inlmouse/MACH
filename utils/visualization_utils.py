import numpy as np
import supervision as sv
from PIL import Image


def draw_predictions(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
) -> Image.Image:
    """
    使用 supervision 在原始图像上绘制检测结果（适配 inference_single 输出）。
    
    参数:
        image (PIL.Image.Image): 
            原始 RGB 图像（来自 inference_single 返回的 orig_image）。
        
        boxes (np.ndarray): 
            边界框坐标，形状 (N, 4)，格式为 [x1, y1, x2, y2]，像素绝对坐标。
            当无检测结果时形状为 (0, 4)。
        
        scores (np.ndarray): 
            置信度分数，形状 (N,)。
        
        labels (np.ndarray): 
            类别 ID，形状 (N,)，元素为整数。
        
        class_names (list[str] | None, optional): 
            类别名称列表。如果提供，则标签显示为 "类别名:置信度"；
            否则显示 "cls{ID}:置信度"。
    
    返回:
        PIL.Image.Image: 绘制好检测框和标签后的图像。
    """
    
    # 转为 numpy（兼容 torch.Tensor 或 list）
    # 如果是 CUDA tensor，先移到 CPU
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu()
    if hasattr(scores, 'cpu'):
        scores = scores.cpu()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu()
    
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    class_ids = np.asarray(labels, dtype=int)

    if len(boxes) == 0:
        return image.copy()   # 返回副本，保持一致行为

    # 创建 Detections 对象
    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids,
    )

    # ==================== 自适应参数（根据图像分辨率） ====================
    resolution_wh = image.size                     # (width, height)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    # ==================== 准备标签文本 ====================
    if class_names is not None:
        labels_list = [
            f"{class_names[cls_id]}:{conf:.2f}" 
            if cls_id < len(class_names) 
            else f"cls{cls_id}:{conf:.2f}"
            for cls_id, conf in zip(detections.class_id, detections.confidence)
        ]
    else:
        labels_list = [
            f"cls{cls_id}:{conf:.2f}"
            for cls_id, conf in zip(detections.class_id, detections.confidence)
        ]

    # ==================== Annotators（使用自适应参数 + PIL 原生支持） ====================
    box_annotator = sv.BoxAnnotator(
        thickness=thickness,
        color_lookup=sv.ColorLookup.CLASS      # 按类别分配颜色
    )

    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_position=sv.Position.TOP_LEFT,
        smart_position=True,                   # 智能避免标签重叠
        color_lookup=sv.ColorLookup.CLASS
    )

    # ==================== 绘制（直接在 PIL Image 上操作） ====================
    annotated_image = image.copy()   # 保护原始图像

    # 先画框，再画标签（顺序很重要）
    annotated_image = box_annotator.annotate(
        scene=annotated_image,
        detections=detections
    )

    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels_list
    )

    return annotated_image