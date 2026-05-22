# ExpAlign: Explicit Alignment for Open-Vocabulary Object Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/pytorch-2.8-%23EE4C2C.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.22666-b31b1b.svg)](https://arxiv.org/abs/2601.22666)

Official implementation for **[ExpAlign: Explicit Alignment for Open-Vocabulary Object Detection with Expectation Alignment Head](https://arxiv.org/abs/2601.22666)**.

## Overview

ExpAlign is a theoretically grounded vision-language alignment framework for open-vocabulary object detection. Built on a principled multiple instance learning formulation, it introduces:

- **Expectation Alignment Head**: Attention-based soft MIL pooling over token-region similarities, enabling implicit token and instance selection without additional annotations
- **Energy-based Multi-scale Consistency**: Top-K multi-positive contrastive objective and Geometry-Aware Consistency derived from Lagrangian-constrained free-energy minimization
- **Dynamic Class Inference**: Runtime class adaptation via `set_class()` for flexible open-vocabulary detection

### Key Results

- **36.2 AP_r** on LVIS minival split (state-of-the-art at comparable model scale)
- Strong performance on long-tail categories
- Lightweight and inference-efficient design

## Installation

### Prerequisites

- Python >= 3.11
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd expalign

# Run setup script
bash scripts/setup_environment.sh
```

Or manually:

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install flash-attn separately (requires compilation)
uv pip install flash-attn --no-build-isolation

# Activate environment
source .venv/bin/activate
```

## Pretrained Weights

### DINOv3 ConvNeXt Backbone

DINOv3 provides ConvNeXt backbones pre-trained on LVD-1689M dataset.

| Model | Parameters | HuggingFace | ModelScope |
|-------|------------|-------------|------------|
| ConvNeXt Tiny | 29M | [Download](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) | [Download](https://modelscope.cn/models/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) |
| ConvNeXt Small | 50M | [Download](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m) | [Download](https://modelscope.cn/models/facebook/dinov3-convnext-small-pretrain-lvd1689m) |
| ConvNeXt Base | 89M | [Download](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m) | [Download](https://modelscope.cn/models/facebook/dinov3-convnext-base-pretrain-lvd1689m) |
| ConvNeXt Large | 198M | [Download](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m) | [Download](https://modelscope.cn/models/facebook/dinov3-convnext-large-pretrain-lvd1689m) |

**Download via HuggingFace CLI:**
```bash
# Install huggingface-hub
pip install huggingface-hub

# Download ConvNeXt Tiny (recommended for most users)
huggingface-cli download facebook/dinov3-convnext-tiny-pretrain-lvd1689m \
    --local-dir ./pretrained/dinov3-convnext-tiny
```

**Download via ModelScope (China mainland):**
```bash
# Install modelscope
pip install modelscope

# Download ConvNeXt Tiny
modelscope download --model facebook/dinov3-convnext-tiny-pretrain-lvd1689m \
    --local_dir ./pretrained/dinov3-convnext-tiny
```

**Note:** Some DINOv3 weights require accepting the license on HuggingFace before download.

### CLIP Text Encoder

We use OpenAI's CLIP ViT-L/14 for text encoding (768-dim embeddings).

| Model | Embedding Dim | Download URL |
|-------|---------------|--------------|
| ViT-L/14 | 768 | [OpenAI Official](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) |
| ViT-L/14@336px | 768 | [OpenAI Official](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt) |

**Download:**
```bash
# Using wget
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt \
    -O ./pretrained/ViT-L-14.pt
```

**Alternative:** Use [OpenCLIP](https://github.com/mlfoundations/open_clip) to download compatible weights:
```bash
pip install open-clip-torch

import open_clip
model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
```

### Qwen3-VL-Embedding Text Encoder

Qwen3-VL-Embedding provides stronger vision-language alignment with higher dimensional embeddings.

| Model | Size | Embedding Dim | HuggingFace | ModelScope |
|-------|------|---------------|-------------|------------|
| Qwen3-VL-Embedding-2B | 2B | 2048 (MRL: 64-2048) | [Download](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | [Download](https://modelscope.cn/models/qwen/Qwen3-VL-Embedding-2B) |
| Qwen3-VL-Embedding-8B | 8B | 4096 (MRL: 64-4096) | [Download](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | [Download](https://modelscope.cn/models/qwen/Qwen3-VL-Embedding-8B) |

**Download via HuggingFace CLI:**
```bash
# Install dependencies
pip install huggingface-hub

# Download 2B model (recommended, supports MRL truncation to 768/1024)
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B \
    --local-dir ./pretrained/Qwen3-VL-Embedding-2B
```

**Download via ModelScope (China mainland):**
```bash
# Install modelscope
pip install modelscope

# Download 2B model
modelscope download --model qwen/Qwen3-VL-Embedding-2B \
    --local_dir ./pretrained/Qwen3-VL-Embedding-2B
```

**Using with ExpAlign:**
```python
# Truncate to 768-dim for compatibility with CLIP-based models
textencoder = Qwen3VLEmbeddingTextEmbedder(
    "./pretrained/Qwen3-VL-Embedding-2B",
    device="cuda",
    mrl_truncate=768  # Truncate 2048 -> 768
)
```

### Model Zoo Summary

| Component | Recommended | Embedding Dim | Size | Download |
|-----------|-------------|---------------|------|----------|
| **Backbone** | DINOv3 ConvNeXt Tiny | - | 29M | [HF](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) / [MS](https://modelscope.cn/models/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) |
| **Text Encoder** | CLIP ViT-L/14 | 768 | 304M | [OpenAI](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) |
| **Text Encoder** | Qwen3-VL-Embedding-2B | 768* | 2B | [HF](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) / [MS](https://modelscope.cn/models/qwen/Qwen3-VL-Embedding-2B) |

*Using MRL truncation from 2048 to 768

## Project Structure

```
.
├── train_ddp.py                # Multi-GPU distributed training
├── test.py                     # Inference and visualization
├── pyproject.toml              # Project dependencies
├── models/
│   ├── convnext.py             # ConvNeXt backbone
│   └── expalignet.py           # Main detection model
├── layers/
│   ├── fpn.py                  # FPN+PAN neck
│   └── head.py                 # Detection head with ExpAlign
├── dataset/
│   ├── unified_dataset.py      # Unified dataset implementation
│   ├── build_dataloader.py     # Dataloader builder
│   ├── transforms.py           # Data augmentation
│   └── textmodelembedder.py    # Text encoder wrappers
├── loss/
│   ├── detectloss.py           # detection loss
│   └── taskalignedassigner.py  # Anchor-to-GT matching
├── utils/
│   ├── train_utils.py          # Training utilities
│   ├── detect_utils.py         # NMS and post-processing
│   └── visualization_utils.py  # Result visualization
└── evaluation/
    ├── coco_evaluator.py       # COCO evaluation
    └── lvis_offline_evaluator.py  # LVIS evaluation
```

## Training (DDP)

### Single-Node Multi-GPU

```bash
torchrun --nproc_per_node=4 train_ddp.py
```

### Multi-Node Multi-GPU

**Node 0 (Master):**
```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr="192.168.1.100" --master_port=12345 train_ddp.py
```

**Node 1:**
```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
    --master_addr="192.168.1.100" --master_port=12345 train_ddp.py
```

### Configuration

Edit `train_ddp.py` to configure training:

```python
###################### Configurations ######################
num_epochs = 30
batch_size_per_gpu = 32       # Batch size per GPU
num_workers = 4               # DataLoader workers per GPU
target_size = 640             # Input image size
output_dir = "outputs-clip"

# Model configuration
text_embed_dim = 768          # Text embedding dimension (CLIP: 768, Qwen: 3584)
backbone_size = "tiny"        # tiny/small/base/large
num_infonce_batch = 80        # Number of contrastive samples per batch
num_classes = num_infonce_batch

# Pretrained weights
pretrain_backbone = "/path/to/dinov3_convnext_tiny_pretrain.pth"
pretrain_text_encoder = "/path/to/ViT-L-14.pt"
# pretrain_text_encoder = "/path/to/Qwen3-VL-Embedding-2B"

# Optimizer
base_lr = 0.002
weight_decay = 0.025
warmup_epochs = 3

# Training
use_amp = True
val_interval = 30             # Validate every N epochs

# Dataset configuration
train_ann_files = [
    "/path/to/Objects365_v1/annotations/objects365_train_segm.json",
    "/path/to/flickr30k/final_flickr_separateGT_train_segm.json",
    "/path/to/MixedGrounding/mdetr_annotations/final_mixed_train_no_coco_segm.json",
]
train_image_roots = [
    "/path/to/Objects365_v1/images/train",
    "/path/to/flickr30k/flickr30k-images",
    "/path/to/MixedGrounding/images",
]

val_ann_files = ["/path/to/coco/annotations/instances_val2017.json"]
val_image_roots = ["/path/to/coco/val2017"]
```

### DDP Features

- **Automatic dataset caching**: Main process builds dataset and caches samples + embeddings; other processes load from cache
- **Synchronized training**: Gradient synchronization across all GPUs
- **Main-process validation**: Evaluation only on rank 0 to avoid redundant computation
- **Automatic checkpointing**: Saves `model_last.pth` (latest) and `model_best.pth` (best mAP)

### Resuming Training

```python
# Set checkpoint path
pretrain_model_path = "outputs-clip/model_last.pth"
```

## Dynamic Class Setting: `set_class()` and `unset_class()`

A core feature of ExpAlign is the ability to dynamically change detection classes at inference time without retraining. This is achieved through the `set_class()` and `unset_class()` mechanisms.


### Usage

#### Basic Inference

```python
import torch
from models.expalignet import expalignet
from dataset.textmodelembedder import CLIPTextEmbedder
from utils.train_utils import load_model

# Load model
device = torch.device("cuda")
model, _, _ = load_model("outputs-clip/model_last.pth", text_embed_dim=768, num_classes=80, device=device)
model.eval()

# Build text encoder
text_encoder = CLIPTextEmbedder("/path/to/ViT-L-14.pt", device=device, mrl_truncate=768)

# Define custom classes
class_names = ["cat", "dog", "person", "car", "bicycle"]

# Encode and set classes
txt_feats = text_encoder.embedtext(class_names, normalize=True)
model.set_class(txt_feats)  # Fuse text features into conv weights

# Inference
with torch.no_grad():
    outputs = model(images)  # Directly outputs class probabilities

# Restore original state (optional)
model.unset_class()
```

#### Open-Vocabulary Detection

```python
# LVIS evaluation: 1203 classes
dataset = LVISDataset(lvis_json, image_root, target_size)

# Encode all LVIS categories
text_feats = text_encoder.embedtext(dataset.category_names, normalize=True)

# Set model to detect all LVIS classes
model.set_class(text_feats)

# Run inference
predictions = run_inference(model, dataloader, device)

# Evaluate with official LVIS API
lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type='bbox')
lvis_eval.run()
```

#### Multiple Class Sets (Sequential Inference)

```python
# First inference: COCO 80 classes
coco_names = [...]  # 80 classes
coco_feats = text_encoder.embedtext(coco_names, normalize=True)
model.set_class(coco_feats)
results_coco = infer(model, images_coco)

# Switch to LVIS 1203 classes
model.unset_class()  # Restore first
lvis_names = [...]  # 1203 classes
lvis_feats = text_encoder.embedtext(lvis_names, normalize=True)
model.set_class(lvis_feats)
results_lvis = infer(model, images_lvis)

# Restore after use
model.unset_class()
```

### Important Notes

1. **Must call `model.eval()` before `set_class()`**: The fusion modifies model parameters and should not be used during training
   
2. **Thread safety**: `set_class()` modifies module state; avoid concurrent calls from multiple threads

3. **Memory**: Fused weights are computed on-the-fly; original parameters are cached for restoration

4. **Multiple calls**: `set_class()` can be called multiple times sequentially; each call saves the original state fresh (supports switching between different class sets)

5. **Token-level head**: For `TokenLevelBNContrastiveHead`, fusion works differently (see `layers/head.py` for implementation details)

## Inference

```bash
python test.py
```

Configure `test.py`:

```python
# Model checkpoint
CKPT_PATH = "outputs-clip/model_last.pth"

# Input (choose one)
IMAGE_PATH = "/path/to/single/image.jpg"
IMAGE_DIR = "/path/to/image/directory"

# Output
OUTPUT_DIR = "test_results"

# Model config
TEXT_EMBED_DIM = 768
TARGET_SIZE = 640
DEVICE = "cuda"

# Inference parameters
CONF_THRESH = 0.25
NMS_THRESH = 0.75

# Class names for visualization
CLASS_NAMES = ['person', 'car', 'dog', 'cat', ...]

# Text encoder
TEXT_ENCODER_TYPE = "clip"  # "clip" or "qwen"
CLIP_MODEL_NAME = "/path/to/ViT-L-14.pt"
QWEN_MODEL_PATH = "/path/to/Qwen3-VL-Embedding-2B"
```

## LVIS Evaluation

```bash
python evaluation/lvis_offline_evaluator.py
```

This runs open-vocabulary evaluation on LVIS dataset using the official LVIS API with `set_class()` for dynamic category adaptation.

## Citation

```bibtex
@article{hu2026expalign,
  title={ExpAlign: Explicit Alignment for Open-Vocabulary Object Detection with Expectation Alignment Head},
  author={Hu, Junyi and others},
  journal={arXiv preprint arXiv:2601.22666},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgements

- [DINOv3](https://github.com/facebookresearch/dinov3) for ConvNeXt pre-trained weights
- [CLIP](https://github.com/openai/CLIP) for text encoding
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) for vision-language embeddings
- [LVIS API](https://github.com/lvis-dataset/lvis-api) for evaluation
