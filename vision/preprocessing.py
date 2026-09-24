"""Image decoding, resizing, and low-level quality metrics.

Quality metrics here are descriptive numbers (blur score, contrast, exposure);
turning them into warnings/pass-fail is `vision.validation`'s job.
"""
from __future__ import annotations

import cv2
import numpy as np


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw uploaded bytes into a BGR image. Raises ValueError if it
    isn't a readable image."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data — unsupported or corrupt file.")
    return img


def resize_for_processing(img: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, float]:
    """Downscale large images for faster processing. Returns (image, scale)
    where scale = processed_size / original_size, so original = processed / scale.
    """
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img, 1.0
    out = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return out, scale


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Lower means blurrier. This is a relative,
    heuristic indicator, not a calibrated sharpness metric."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def contrast_score(gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities. Low values indicate a flat,
    low-contrast image where edges will be hard to segment reliably."""
    return float(gray.std())


def exposure_fractions(gray: np.ndarray) -> tuple[float, float]:
    """Fraction of pixels that are near-black / near-white (clipped)."""
    total = gray.size
    dark = float(np.sum(gray <= 5)) / total
    bright = float(np.sum(gray >= 250)) / total
    return dark, bright
