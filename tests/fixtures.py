"""Synthetic test-image generators with known ground truth.

There's no real camera/part available in this environment, so correctness is
verified against images with an exactly known geometry instead of real photos.
"""
from __future__ import annotations

import cv2
import numpy as np


def make_axis_aligned_part(
    canvas_hw: tuple[int, int] = (600, 800),
    rect_xywh: tuple[int, int, int, int] = (250, 200, 300, 200),
    holes: list[tuple[int, int, int]] | None = None,
    bg_value: int = 40,
    fg_value: int = 210,
) -> tuple[np.ndarray, dict]:
    """A solid rectangle on a plain background, with optional circular holes
    (cx, cy, r in absolute canvas pixel coordinates) cut out to background value."""
    H, W = canvas_hw
    gray = np.full((H, W), bg_value, dtype=np.uint8)
    x, y, w, h = rect_xywh
    gray[y:y + h, x:x + w] = fg_value

    holes = holes or []
    for cx, cy, r in holes:
        cv2.circle(gray, (cx, cy), r, bg_value, -1)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img, {"bbox": rect_xywh, "holes": holes}


def make_rounded_rect_part(
    canvas_hw: tuple[int, int] = (600, 800),
    rect_xywh: tuple[int, int, int, int] = (250, 200, 300, 200),
    corner_radius: int = 30,
    bg_value: int = 40,
    fg_value: int = 210,
) -> tuple[np.ndarray, dict]:
    H, W = canvas_hw
    gray = np.full((H, W), bg_value, dtype=np.uint8)
    x, y, w, h = rect_xywh
    r = corner_radius

    cv2.rectangle(gray, (x + r, y), (x + w - r, y + h), fg_value, -1)
    cv2.rectangle(gray, (x, y + r), (x + w, y + h - r), fg_value, -1)
    for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + w - r, y + h - r), (x + r, y + h - r)]:
        cv2.circle(gray, (cx, cy), r, fg_value, -1)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img, {"bbox": rect_xywh, "corner_radius": corner_radius}


def make_checkerboard_image(
    square_px: int = 40, cols_inner: int = 7, rows_inner: int = 5, margin: int = 60,
) -> np.ndarray:
    """`cols_inner`/`rows_inner` = inner-corner counts, matching what
    `cv2.findChessboardCorners` expects as `pattern_size`."""
    squares_x, squares_y = cols_inner + 1, rows_inner + 1
    board_w, board_h = squares_x * square_px, squares_y * square_px

    board = np.zeros((board_h, board_w), dtype=np.uint8)
    for i in range(squares_y):
        for j in range(squares_x):
            if (i + j) % 2 == 0:
                board[i * square_px:(i + 1) * square_px, j * square_px:(j + 1) * square_px] = 255

    canvas = np.full((board_h + 2 * margin, board_w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin:margin + board_h, margin:margin + board_w] = board
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
