# StyleGAN3-R Fine-tuning on CelebV-HQ Video Frames

## Overview

This repository contains the training and inference code used for the Deep Generative Models final project.

The project fine-tunes **StyleGAN3-R** on face images extracted from the CelebV-HQ video dataset and evaluates the effects of frame sampling strategies and training length on image generation quality.

## Environment

The implementation follows the official StyleGAN3 framework.

### Hardware

* GPU: NVIDIA RTX 3090
* CUDA: 11.8

### Software

* Python 3.8
* PyTorch
* StyleGAN3

---

## Dataset Preparation

Raw videos are stored in:

```bash
/ssdg/spl_huiwon/generation/stylegan3/datasets/celebvhq_raw
```

The StyleGAN3 dataset zip file is generated using:

```bash
/ssdg/spl_huiwon/generation/stylegan3/dataset_tool.py
```

### One Frame per Video

```bash
python dataset_tool.py \
    --source=/ssdg/spl_huiwon/generation/stylegan3/datasets/celebvhq_raw \
    --dest=/ssdg/spl_huiwon/generation/stylegan3/datasets/celebvhq-256x256.zip \
    --resolution=256x256 \
    --fps=1 \
    --max-frames-per-video=1
```

### Three Frames per Video

For the main experiment, three frames were extracted from each video at approximately 25%, 50%, and 75% of the video duration.

```bash
python dataset_tool.py \
    --source=/ssdg/spl_huiwon/generation/stylegan3/datasets/celebvhq_raw \
    --dest=/ssdg/spl_huiwon/generation/stylegan3/datasets/celebvhq-256x256.zip \
    --resolution=256x256 \
    --fps=1 \
    --max-frames-per-video=3
```

---

## Training

Training is performed using:

```bash
/ssdg/spl_huiwon/generation/stylegan3/train.py
```

The model is initialized from the official pretrained checkpoint:

```bash
/ssdg/spl_huiwon/generation/stylegan3/stylegan3-r-ffhqu-256x256.pkl
```

Training command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --outdir=/ssdg/spl_huiwon/generation/stylegan3/training-runs \
    --cfg=stylegan3-r \
    --data=/ssdg/spl_huiwon/generation/stylegan3/datasets/celebvhq-256x256.zip \
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
    --resume=/ssdg/spl_huiwon/generation/stylegan3/stylegan3-r-ffhqu-256x256.pkl
```

---

## Inference

Download the trained checkpoint:

```text
network-snapshot-005000.pkl
```

and place it in the StyleGAN3 directory.

Image generation is performed using:

```bash
python gen_images.py \
    --network=network-snapshot-005000.pkl \
    --seeds=0-999 \
    --trunc=1 \
    --noise-mode=const \
    --outdir=out
```

---

## Seed

The final submitted images were generated using:

```bash
--seeds=0-999
```

---

## Notes

* Model weights are not included in this repository.
* The repository only contains code and instructions required for reproducibility.
* The best-performing model was obtained using three frames per video and a training length of 4000 kimg.
