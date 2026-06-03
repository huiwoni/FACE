import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import click
import PIL.Image
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def error(msg: str):
    print("Error:", msg)
    sys.exit(1)


def parse_tuple(s: str) -> Tuple[int, int]:
    s = s.lower().replace(",", "x")
    parts = s.split("x")
    if len(parts) != 2:
        raise ValueError(f"cannot parse resolution: {s}")
    return int(parts[0]), int(parts[1])


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        error("ffmpeg not found. Install with: conda install -y -c conda-forge ffmpeg")


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def run_ffmpeg_extract(
    video_path: Path,
    frame_dir: Path,
    fps: float,
    resolution: Tuple[int, int],
):
    frame_dir.mkdir(parents=True, exist_ok=True)
    width, height = resolution

    vf = (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        "-vf", vf,
        str(frame_dir / "frame_%08d.png"),
    ]

    subprocess.run(cmd, check=True)


def write_frames_to_zip(
    frame_dir: Path,
    zf: zipfile.ZipFile,
    start_idx: int,
    max_images: Optional[int],
    max_frames_per_video: Optional[int],
) -> int:
    frame_paths = sorted(frame_dir.glob("*.png"))
    written = 0

    for frame_path in frame_paths:
        if max_frames_per_video is not None and written >= max_frames_per_video:
            break
        if max_images is not None and start_idx + written >= max_images:
            break

        idx = start_idx + written
        idx_str = f"{idx:08d}"
        archive_fname = f"{idx_str[:5]}/img{idx_str}.png"

        img = PIL.Image.open(frame_path).convert("RGB")

        image_bits = io.BytesIO()
        img.save(image_bits, format="png", compress_level=0, optimize=False)
        zf.writestr(archive_fname, image_bits.getbuffer())

        written += 1

    return written


def collect_videos_from_dir(source_dir: Path):
    return sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def process_video_file(
    video_path: Path,
    zf: zipfile.ZipFile,
    global_idx: int,
    fps: float,
    resolution: Tuple[int, int],
    max_images: Optional[int],
    max_frames_per_video: Optional[int],
    tmp_root: Path,
) -> int:
    frame_dir = tmp_root / f"frames_{global_idx:08d}"

    try:
        run_ffmpeg_extract(video_path, frame_dir, fps=fps, resolution=resolution)
        written = write_frames_to_zip(
            frame_dir=frame_dir,
            zf=zf,
            start_idx=global_idx,
            max_images=max_images,
            max_frames_per_video=max_frames_per_video,
        )
    finally:
        if frame_dir.exists():
            shutil.rmtree(frame_dir)

    return written


@click.command()
@click.option("--source", required=True, type=str, help="Input path: .tar.gz containing videos, video directory, or single video file")
@click.option("--dest", required=True, type=str, help="Output StyleGAN dataset zip")
@click.option("--resolution", default="256x256", type=parse_tuple, show_default=True, help="Output resolution")
@click.option("--fps", default=1.0, type=float, show_default=True, help="Frames per second to extract from each video")
@click.option("--max-images", default=None, type=int, help="Maximum total number of extracted images")
@click.option("--max-frames-per-video", default=None, type=int, help="Maximum frames extracted per video")
def main(
    source: str,
    dest: str,
    resolution: Tuple[int, int],
    fps: float,
    max_images: Optional[int],
    max_frames_per_video: Optional[int],
):
    check_ffmpeg()
    PIL.Image.init()

    source_path = Path(source)
    dest_path = Path(dest)

    if not source_path.exists():
        error(f"source does not exist: {source}")

    if dest_path.exists():
        error(f"destination already exists. Remove it first: {dest}")

    if dest_path.parent:
        dest_path.parent.mkdir(parents=True, exist_ok=True)

    global_idx = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_STORED) as zf:
            if source_path.is_dir():
                videos = collect_videos_from_dir(source_path)
                if len(videos) == 0:
                    error(f"no video files found in directory: {source}")

                for video_path in tqdm(videos, desc="videos"):
                    if max_images is not None and global_idx >= max_images:
                        break

                    written = process_video_file(
                        video_path=video_path,
                        zf=zf,
                        global_idx=global_idx,
                        fps=fps,
                        resolution=resolution,
                        max_images=max_images,
                        max_frames_per_video=max_frames_per_video,
                        tmp_root=tmp_root,
                    )
                    global_idx += written

            elif source_path.is_file() and is_video(str(source_path)):
                written = process_video_file(
                    video_path=source_path,
                    zf=zf,
                    global_idx=global_idx,
                    fps=fps,
                    resolution=resolution,
                    max_images=max_images,
                    max_frames_per_video=max_frames_per_video,
                    tmp_root=tmp_root,
                )
                global_idx += written

            elif source_path.is_file() and tarfile.is_tarfile(source_path):
                with tarfile.open(source_path, "r:*") as tar:
                    members = [
                        m for m in tar.getmembers()
                        if m.isfile() and Path(m.name).suffix.lower() in VIDEO_EXTS
                    ]

                    if len(members) == 0:
                        error(f"no video files found in tar archive: {source}")

                    for vid_idx, member in enumerate(tqdm(members, desc="videos")):
                        if max_images is not None and global_idx >= max_images:
                            break

                        suffix = Path(member.name).suffix.lower()
                        tmp_video = tmp_root / f"video_{vid_idx:08d}{suffix}"

                        with tar.extractfile(member) as src_f:
                            if src_f is None:
                                continue
                            with open(tmp_video, "wb") as dst_f:
                                shutil.copyfileobj(src_f, dst_f)

                        written = process_video_file(
                            video_path=tmp_video,
                            zf=zf,
                            global_idx=global_idx,
                            fps=fps,
                            resolution=resolution,
                            max_images=max_images,
                            max_frames_per_video=max_frames_per_video,
                            tmp_root=tmp_root,
                        )
                        global_idx += written

                        tmp_video.unlink(missing_ok=True)

            else:
                error("source must be a video directory, a single video file, or a tar/tar.gz archive containing videos")

            metadata = {"labels": None}
            zf.writestr("dataset.json", json.dumps(metadata))

    print(f"Done. Wrote {global_idx} images to {dest_path}")


if __name__ == "__main__":
    main()
