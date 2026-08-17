# Single-image / batch inference + visualization script (see the Inference section in README)
# Usage: edit the CONFIG section below, then run  uv run python test.py

import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from dataset.transforms import make_coco_transforms
from dataset.textmodelembedder import Qwen3VLEmbeddingTextEmbedder, CLIPTextEmbedder
from utils.detr_train_utils import load_model
from utils.visualization_utils import draw_predictions


# ==================== CONFIG ====================
# Edit the following settings for your test case

CKPT_PATH = "model_ach-woffn-rmjepa-epoch30.pth"   # Path to model checkpoint
IMAGE_PATH = "test3-squ.jpg"     # Single image path (or a list of paths)
IMAGE_DIR = None                              # Image directory (used when IMAGE_PATH is None)
OUTPUT_DIR = "test_results"                   # Directory for visualization results
MAX_IMAGES = None                             # Max number of images to process, None = all

# Model settings (must match the checkpoint's training configuration)
MODEL_NAME = "expalignet"   # "expalignet" (CNN detector) or "vlrtdetrnet" (DETR variant)
BACKBONE_SIZE = "tiny"      # tiny/small/base
TEXT_EMBED_DIM = 768        # Text embedding dim (Tiny/Small: 768, Base: 1024)
TARGET_SIZE = 640
DEVICE = "cuda:0"

# Inference settings
CONF_THRESH = 0.05
NMS_THRESH = 0.75

# Detection classes: arbitrary natural-language referring expressions are supported
CLASS_NAMES = [
    'the right shoe', 'knee-high socks', 'pleated skirt',
    'school bag', 'person', 'bangs', 'pony tail', 'train', 'manhole cover',
]

# Text encoder settings
TEXT_ENCODER_TYPE = "qwen"  # "qwen" or "clip"
QWEN_MODEL_PATH = "/path/to/Qwen3-VL-Embedding-2B"
CLIP_MODEL_NAME = "pretrained/ViT-L-14.pt"

# ==================== Helper functions ====================

def preprocess_image(image_path, target_size=640):
    """Preprocess a single image"""
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    transforms = make_coco_transforms(istrain=False, target_size=target_size)
    image_tensor, _ = transforms(image, None)
    return image_tensor, (orig_w, orig_h), image


def inference_single(model, image_path, text_feats=None, mask=None, device='cuda', target_size=640,
                     conf_thresh=0.25, nms_thresh=0.5):
    """
    Single-image inference (via the model's built-in predict: dynamic class mounting,
    NMS, and coordinate mapping back to the original resolution)
    """
    # Preprocess
    image_tensor, orig_size, orig_image = preprocess_image(image_path, target_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    orig_h, orig_w = orig_size[1], orig_size[0]

    orig_target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
    results = model.predict(
        x=image_tensor,
        w=text_feats,
        m=mask,
        conf_threshold=conf_thresh,
        nms_threshold=nms_thresh,
        orig_target_sizes=orig_target_sizes,
    )
    r = results[0]
    return orig_image, r["boxes"], r["scores"], r["labels"]


def get_test_images(image_path, image_dir, max_images=None):
    """Collect the list of test images"""
    if isinstance(image_path, list):
        return image_path[:max_images] if max_images else image_path
    if isinstance(image_path, str):
        return [image_path]
    if image_dir:
        image_paths = (
            list(Path(image_dir).glob('*.jpg')) +
            list(Path(image_dir).glob('*.png')) +
            list(Path(image_dir).glob('*.jpeg'))
        )
        if max_images:
            image_paths = image_paths[:max_images]
        return image_paths

    raise ValueError("Please configure IMAGE_PATH or IMAGE_DIR")


def build_text_encoder(encoder_type, embed_dim, device):
    """Build the text encoder"""
    if encoder_type == "qwen":
        return Qwen3VLEmbeddingTextEmbedder(
            QWEN_MODEL_PATH,
            device=device,
            mrl_truncate=embed_dim
        )
    elif encoder_type == "clip":
        return CLIPTextEmbedder(
            CLIP_MODEL_NAME,
            device=device,
            mrl_truncate=embed_dim
        )
    else:
        raise ValueError(f"Unsupported text encoder type: {encoder_type}")


# ==================== Main ====================

def main():
    # 1. Prepare output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load model (factory mode; class count is inferred from the checkpoint)
    print(f"Loading model: {CKPT_PATH} ({MODEL_NAME})")
    model, _, _ = load_model(
        ckpt_path=CKPT_PATH,
        args={
            'model_name': MODEL_NAME,
            'text_embed_dim': TEXT_EMBED_DIM,
            'size': BACKBONE_SIZE,
            'num_classes': None,    # None = infer from checkpoint weights
            'reg_max': 16,
        },
        device=DEVICE,
        resume=False,   # inference only loads weights, no optimizer state
    )
    model.eval()
    print(f"Model loaded. Classes: {len(CLASS_NAMES)}, device: {DEVICE}")

    # 3. Load the text encoder and encode CLASS_NAMES into token-level features
    text_encoder = build_text_encoder(TEXT_ENCODER_TYPE, TEXT_EMBED_DIM, DEVICE)
    text_feats, mask = text_encoder.embedtext(CLASS_NAMES, normalize=True, tokenlevel=True)

    # 4. Collect test images
    image_paths = get_test_images(IMAGE_PATH, IMAGE_DIR, MAX_IMAGES)
    print(f"Found {len(image_paths)} test images")

    # 5. Run inference and visualize one by one
    for img_path in tqdm(image_paths, desc="Inferring"):
        try:
            image, boxes, scores, labels = inference_single(
                model, str(img_path), text_feats, mask, DEVICE, TARGET_SIZE,
                CONF_THRESH, NMS_THRESH
            )
            print(f"{img_path}: {len(boxes)} objects detected")

            vis_image = draw_predictions(
                image, boxes, scores, labels,
                class_names=CLASS_NAMES,
            )

            save_name = Path(img_path).stem + '_result.jpg'
            save_path = os.path.join(OUTPUT_DIR, save_name)
            vis_image.save(save_path)

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
