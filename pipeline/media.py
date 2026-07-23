"""Media preparation: turn an uploaded image or short video into a small set of
good frames, plus a cheap liveness/motion signal.

Design goals for the PoC:
- No optical-flow / homography stabilization (cut from the docs). Just sample frames,
  keep the sharpest, and measure how much the scene actually moves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np


@dataclass
class PreparedMedia:
    frame_paths: List[str] = field(default_factory=list)  # temp files for the VLM
    is_video: bool = False
    motion_score: float = 0.0   # mean inter-frame difference, 0..~255
    sharpness: float = 0.0      # variance of Laplacian of the best frame


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _save(frame: np.ndarray, out_dir: str, idx: int) -> str:
    path = os.path.join(out_dir, f"frame_{idx:02d}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def prepare_image(image_path: str, out_dir: str) -> PreparedMedia:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = _save(img, out_dir, 0)
    return PreparedMedia(frame_paths=[out], is_video=False, motion_score=0.0,
                         sharpness=_sharpness(gray))


def prepare_video(video_path: str, out_dir: str, sample_k: int = 12,
                  keep_n: int = 4) -> PreparedMedia:
    """Sample up to sample_k frames evenly, keep the keep_n sharpest, and compute
    a motion score from consecutive sampled frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        # Fallback: read sequentially.
        frames = []
        while len(frames) < sample_k:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    else:
        idxs = np.linspace(0, max(total - 1, 0), num=min(sample_k, total)).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
    cap.release()

    if not frames:
        raise ValueError("No frames could be read from the video.")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # Motion score: mean absolute difference between consecutive sampled frames.
    diffs = []
    for a, b in zip(grays, grays[1:]):
        if a.shape == b.shape:
            diffs.append(float(np.mean(cv2.absdiff(a, b))))
    motion = float(np.mean(diffs)) if diffs else 0.0

    # Keep the sharpest keep_n frames, preserving temporal order.
    scored = sorted(range(len(frames)), key=lambda i: _sharpness(grays[i]), reverse=True)
    keep = sorted(scored[:keep_n])
    paths = [_save(frames[i], out_dir, j) for j, i in enumerate(keep)]
    best_sharp = _sharpness(grays[scored[0]]) if scored else 0.0

    return PreparedMedia(frame_paths=paths, is_video=True,
                         motion_score=motion, sharpness=best_sharp)


def prepare(media_path: str, out_dir: str) -> PreparedMedia:
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(media_path)[1].lower()
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    if ext in video_exts:
        return prepare_video(media_path, out_dir)
    return prepare_image(media_path, out_dir)
