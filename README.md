# StyleGAN3-R Fine-tuning on CelebV-HQ Video Frames

## Overview

This repository contains the training and inference code used for the AI 518 Deep Generative Models final project.

The project fine-tunes **StyleGAN3-R** on face images extracted from the CelebV-HQ video dataset and evaluates the effects of frame sampling strategies and training length on image generation quality.

## Environment Setup

This project is based on the official StyleGAN3 implementation:

* [StyleGAN3 Model Zoo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files?version=1)

Clone the repository:

```bash
git clone https://github.com/NVlabs/stylegan3.git
cd stylegan3
```

Create a conda environment:

```bash
conda create -n stylegan3 python=3.8 -y
conda activate stylegan3
```

Install PyTorch (CUDA 11.8 example):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install StyleGAN3 dependencies:

```bash
pip install click requests tqdm pyspng ninja imageio imageio-ffmpeg scipy psutil
```

Experiments were conducted using:

* GPU: NVIDIA RTX 3090
* CUDA: 11.8
* Python: 3.8

---

## Dataset Preparation

Raw videos are stored in:

```text
datasets/celebvhq_raw
```

The StyleGAN3 dataset zip file is generated using:

```bash
dataset_tool.py
```

### One Frame per Video

```bash
python dataset_tool.py \
    --source=datasets/celebvhq_raw \
    --dest=datasets/celebvhq-256x256.zip \
    --resolution=256x256 \
    --fps=1 \
    --max-frames-per-video=1
```

### Three Frames per Video

For the main experiment, three frames were extracted from each video at approximately **25%, 50%, and 75%** of the video duration.

```bash
python dataset_tool.py \
    --source=datasets/celebvhq_raw \
    --dest=datasets/celebvhq-256x256.zip \
    --resolution=256x256 \
    --fps=1 \
    --max-frames-per-video=3
```

> The frame extraction logic was modified to sample frames from the 25%, 50%, and 75% temporal positions of each video instead of simply taking the first three frames.

---

## Training

Training is performed using:

```bash
train.py
```

The model is initialized from the official StyleGAN3-R FFHQ-U pretrained checkpoint.

### Pretrained Model

The pretrained model can be downloaded from:

* [Pretrained Model Download (Google Drive)]((https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files?version=1))

Expected checkpoint:

```text
stylegan3-r-ffhqu-256x256.pkl
```

Training command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --outdir=training-runs \
    --cfg=stylegan3-r \
    --data=datasets/celebvhq-256x256.zip \
    --gpus=1 \
    --batch=32 \
    --gamma=2 \
    --mirror=1 \
    --aug=ada \
    --target=0.6 \
    --kimg=5000 \
    --tick=1 \
    --snap=50 \
    --metrics=none \
    --glr=0.0025 \
    --dlr=0.001 \
    --seed=0 \
    --resume=stylegan3-r-ffhqu-256x256.pkl
```

### Main Training Configuration

| Item             | Value       |
| ---------------- | ----------- |
| Model            | StyleGAN3-R |
| Resolution       | 256 × 256   |
| Batch Size       | 32          |
| GPU              | RTX 3090    |
| R1 Gamma         | 2           |
| ADA              | Enabled     |
| ADA Target       | 0.6         |
| Generator LR     | 0.0025      |
| Discriminator LR | 0.001       |
| Seed             | 0           |
| Total Training   | 5000 kimg   |

---

## Inference

Image generation is performed using:

```bash
gen_images.py
```

Download the trained checkpoint:

* [Trained Checkpoint Download (Google Drive)](https://drive.google.com/drive/folders/1ZDQSmP5WOmhKGnm3ZqxBMsGN9nXp7udN?utm_source=chatgpt.com)

Expected checkpoint:

```text
network-snapshot-005000.pkl
```

Generate images:

```bash
python gen_images.py \
    --network=network-snapshot-005000.pkl \
    --seeds=0-999 \
    --trunc=1 \
    --noise-mode=const \
    --outdir=out
```

---

## Reproducibility

Final submission images were generated using:

```bash
--seeds=0-999
```

Random seed used during training:

```bash
--seed=0
```

---

## Notes

* Model weights are not included in this repository.
* Pretrained checkpoints and trained model checkpoints can be downloaded from the provided Google Drive link.
* The best-performing model was obtained using three frames per video and a training length of 4000 kimg.
* This repository contains only the code and commands necessary for reproducing the reported results.
