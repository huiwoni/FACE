# StyleGAN3-R Fine-tuning on CelebV-HQ Video Frames

## Environment Setup

This project is based on the official StyleGAN3 implementation.

* [StyleGAN3 Official Repository](https://github.com/NVlabs/stylegan3)
* [NVIDIA StyleGAN3 Model Zoo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files?version=1)

### Hardware

* NVIDIA RTX 3090
* CUDA 11.8

### Installation

```bash
git clone https://github.com/NVlabs/stylegan3.git
cd stylegan3

conda create -n stylegan3 python=3.8 -y
conda activate stylegan3

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install click requests tqdm pyspng ninja imageio imageio-ffmpeg scipy psutil
```

---

## Dataset Preparation

Raw CelebV-HQ videos should be placed in:

```text
datasets/celebvhq_raw
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

### Three Frames per Video (Main Experiment)

Frames are sampled at approximately 25%, 50%, and 75% of the video duration.

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

Download the pretrained StyleGAN3-R checkpoint from:

* [NVIDIA StyleGAN3 Model Zoo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files?version=1)

Checkpoint:

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

## Testing / Image Generation

Download the trained checkpoint from:

* [Project Checkpoints](https://drive.google.com/drive/folders/1ZDQSmP5WOmhKGnm3ZqxBMsGN9nXp7udN)

Checkpoint:

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

### Reproducibility

Training seed:

```bash
--seed=0
```

Generation seeds:

```bash
--seeds=0-999
```
