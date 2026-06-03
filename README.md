# StyleGAN3-R Fine-tuning on CelebV-HQ Video Frames

## Environment Setup

This project is based on the official StyleGAN3 implementation.

Please follow the environment setup instructions in the [official StyleGAN3 repository](https://github.com/NVlabs/stylegan3).

The experiments were conducted using:

- GPU: NVIDIA RTX 3090
- CUDA: 12.8

---

## Data Preparation

Raw CelebV-HQ videos should be placed in:

```text
datasets/celebvhq_raw
```

The StyleGAN3 dataset zip file is generated using `dataset_tool.py`.

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

For the main experiment, three frames were extracted from each video at 25%, 50%, and 75% of the video duration.

```bash
python dataset_tool.py \
    --source=datasets/celebvhq_raw \
    --dest=datasets/celebvhq-256x256.zip \
    --resolution=256x256 \
    --fps=1 \
    --max-frames-per-video=3
```

---

## Training

The model was initialized from the official StyleGAN3-R FFHQ-U pretrained checkpoint.

The pretrained checkpoint can be downloaded from the [NVIDIA StyleGAN3 Model Zoo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files?version=1).

Checkpoint used for fine-tuning:

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

---

## Test / Image Generation

The trained checkpoint can be downloaded from [Project Checkpoints](https://drive.google.com/drive/folders/1ZDQSmP5WOmhKGnm3ZqxBMsGN9nXp7udN).

Checkpoint used for image generation:

```text
network-snapshot-005000.pkl
```

Image generation command:

```bash
python gen_images.py \
    --network=network-snapshot-005000.pkl \
    --seeds=0-999 \
    --trunc=1 \
    --noise-mode=const \
    --outdir=out
```
