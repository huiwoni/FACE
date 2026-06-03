# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle.

Modified for submission format:

submission.zip
|-- img_0000.jpg
|-- img_0001.jpg
|-- ...
|-- img_0999.jpg
"""

import os
import re
import zipfile
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    """
    if isinstance(s, list):
        return s

    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')

    for p in s.split(','):
        m = range_re.match(p)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            ranges.extend(range(start, end + 1))
        else:
            ranges.append(int(p))

    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse a floating point 2-vector of syntax 'a,b'.

    Example: '0,1' returns (0, 1)
    """
    if isinstance(s, tuple):
        return s

    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))

    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)

    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)

    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]

    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]

    return m

#----------------------------------------------------------------------------

def make_submission_zip(outdir: str, zip_name: str):
    """Create a flat submission zip.

    The zip will contain:
    img_0000.jpg
    img_0001.jpg
    ...
    """
    image_files = sorted(
        f for f in os.listdir(outdir)
        if f.lower().endswith(".jpg") and f.startswith("img_")
    )

    if len(image_files) == 0:
        raise click.ClickException(f"No jpg images found in {outdir}")

    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in image_files:
            full_path = os.path.join(outdir, fname)
            zf.write(full_path, arcname=fname)

    print(f'Created "{zip_name}" with {len(image_files)} images.')

#----------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds, e.g., "0,1,4-6"', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label, unconditional if not specified')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate, e.g., "0.3,1"', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--zip-name', help='Name of submission zip file. Use empty string to skip zip creation.', type=str, default='submission.zip', show_default=True)
@click.option('--jpeg-quality', help='JPEG quality', type=int, default=95, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float, float],
    rotate: float,
    class_idx: Optional[int],
    zip_name: str,
    jpeg_quality: int,
):
    """Generate images using pretrained network pickle.

    Example:

    python gen_images.py \
        --outdir=submission_imgs \
        --trunc=1 \
        --seeds=0-999 \
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    """

    print(f'Loading networks from "{network_pkl}"...')

    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print(f'Generating image for seed {seed} ({seed_idx + 1}/{len(seeds)}) ...')

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        # Construct an inverse rotation/translation matrix and pass to the generator.
        # The generator expects this matrix as an inverse to avoid potentially failing
        # numerical operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # Submission filename format:
        # img_0000.jpg, img_0001.jpg, ..., img_0999.jpg
        save_path = os.path.join(outdir, f'img_{seed_idx:04d}.jpg')
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(save_path, quality=jpeg_quality)

    # Make submission.zip with flat structure.
    if zip_name.strip() != "":
        make_submission_zip(outdir, zip_name)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
