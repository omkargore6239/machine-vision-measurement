"""Synthetic test-image generators with known ground truth.

There's no real camera/part available in this environment, so correctness is
verified against images with an exactly known geometry instead of real photos.
"""
from __future__ import annotations

import math

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


def add_printed_text(
    img: np.ndarray, rect_xywh: tuple[int, int, int, int],
    text: str = "MODEL 8080 QD@", ink_offset: int = -70,
    font_scale: float = 0.9, thickness: int = 2,
) -> np.ndarray:
    """Draws stroke text (simulating printed/engraved product labeling) near
    the top of the part rectangle. Several characters ("O", "0", "Q", "D",
    "8", "@") have closed round loops — a deliberately adversarial choice,
    since a loop's OUTER boundary alone can look deceptively circular/solid
    to a naive detector (see vision.geometry's contrast check, which is what
    actually catches this: the loop's "interior" is mostly surrounding
    material color, not a real cavity)."""
    x, y, w, h = rect_xywh
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    material_value = int(gray[y + 15, x + w // 2])
    ink_value = int(np.clip(material_value + ink_offset, 0, 255))
    out = img.copy()
    cv2.putText(out, text, (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (ink_value, ink_value, ink_value), thickness, cv2.LINE_8)
    return out


def add_texture_noise(
    img: np.ndarray, rect_xywh: tuple[int, int, int, int],
    n_specks: int = 250, max_radius: int = 3, min_intensity_delta: int = 10,
    max_intensity_delta: int = 35, avoid_regions: list[tuple[int, int, int]] | None = None,
    y_offset_from_top: int = 60, seed: int = 0,
) -> np.ndarray:
    """Scatters small random-intensity specks inside the part rectangle
    (below `y_offset_from_top`, to stay clear of any printed text) to
    simulate texture, grain, or compression artifacts. Skips any speck that
    would land inside/near an `avoid_regions` circle (cx, cy, r), so a real
    hole's own boundary isn't accidentally corrupted by the noise."""
    rng = np.random.RandomState(seed)
    x, y, w, h = rect_xywh
    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avoid_regions = avoid_regions or []

    placed = 0
    attempts = 0
    while placed < n_specks and attempts < n_specks * 20:
        attempts += 1
        cx = rng.randint(x + 10, x + w - 10)
        cy = rng.randint(y + y_offset_from_top, y + h - 10)
        if any(math.hypot(cx - ax, cy - ay) < ar + 15 for ax, ay, ar in avoid_regions):
            continue
        r = rng.randint(1, max_radius + 1)
        material_value = int(gray[cy, cx])
        delta = rng.choice([-1, 1]) * rng.randint(min_intensity_delta, max_intensity_delta + 1)
        value = int(np.clip(material_value + delta, 0, 255))
        cv2.circle(out, (cx, cy), r, (value, value, value), -1)
        placed += 1

    return out


def make_circular_flange_part(
    canvas_hw: tuple[int, int] = (700, 700), center: tuple[int, int] = (350, 350),
    outer_r: int = 250, bolt_circle_r: int = 180, n_bolt_holes: int = 6,
    bolt_hole_r: int = 14, center_bore_r: int = 40,
    bg_value: int = 30, fg_value: int = 190,
) -> tuple[np.ndarray, dict]:
    """A non-rectangular (circular flange) part with holes on a bolt circle
    plus a center bore — covers the "circular flange" and "multiple holes"
    part types the app must handle without being rectangle-specific."""
    H, W = canvas_hw
    gray = np.full((H, W), bg_value, dtype=np.uint8)
    cx, cy = center
    cv2.circle(gray, (cx, cy), outer_r, fg_value, -1)

    holes: list[tuple[int, int, int]] = []
    for i in range(n_bolt_holes):
        angle = 2 * math.pi * i / n_bolt_holes
        hx = int(cx + bolt_circle_r * math.cos(angle))
        hy = int(cy + bolt_circle_r * math.sin(angle))
        cv2.circle(gray, (hx, hy), bolt_hole_r, bg_value, -1)
        holes.append((hx, hy, bolt_hole_r))

    cv2.circle(gray, (cx, cy), center_bore_r, bg_value, -1)
    holes.append((cx, cy, center_bore_r))

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img, {"holes": holes, "outer_r": outer_r, "center": center}


def make_busy_product_photo(
    canvas_hw: tuple[int, int] = (900, 1200),
    rect_xywh: tuple[int, int, int, int] = (200, 150, 800, 600),
    real_holes: list[tuple[int, int, int]] | None = None,
    bg_value: int = 25, fg_value: int = 180, seed: int = 1,
) -> tuple[np.ndarray, dict]:
    """A deliberately adversarial synthetic product photo combining several
    historically common false-positive sources in one image: printed model
    text, a vent grille of thin parallel slots, a barcode-like block of
    bars, and fine texture/grain — plus a handful of real holes to verify
    those still survive all of it. The closest reproduction available here
    of the kind of busy real photo that produced mass false positives,
    since no real camera photo was available for this fix."""
    real_holes = real_holes if real_holes is not None else []
    img, gt = make_axis_aligned_part(canvas_hw, rect_xywh, holes=real_holes, bg_value=bg_value, fg_value=fg_value)
    x, y, w, h = rect_xywh

    img = add_printed_text(img, rect_xywh, text="ACME MODEL X-8080 QD@ REV2", ink_offset=-80, font_scale=1.1, thickness=2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vent_x0, vent_y0 = x + 40, y + h - 140
    for i in range(14):
        sx = vent_x0 + i * 18
        cv2.rectangle(gray, (sx, vent_y0), (sx + 6, vent_y0 + 90), min(255, bg_value + 15), -1)

    rng = np.random.RandomState(seed)
    bx0, by0 = x + w - 220, y + 30
    px = bx0
    while px < bx0 + 180:
        bar_w = rng.randint(2, 6)
        if rng.rand() > 0.5:
            cv2.rectangle(gray, (px, by0), (px + bar_w, by0 + 60), 10, -1)
        px += bar_w + rng.randint(2, 5)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = add_texture_noise(img, rect_xywh, n_specks=400, avoid_regions=real_holes, y_offset_from_top=170, seed=seed)

    return img, {"bbox": rect_xywh, "holes": real_holes}


def make_c_shape_part(
    canvas_hw: tuple[int, int] = (700, 700), center: tuple[int, int] = (350, 350),
    arc_radius: int = 200, arm_width: int = 70,
    start_angle: float = 200, end_angle: float = 340,
    bg_value: int = 30, fg_value: int = 190,
) -> tuple[np.ndarray, dict]:
    """An open/concave C- or U-bracket shape — a thick partial ring with a
    genuine gap, like the real forged component this fixture was built
    against. Deliberately has a LOW bounding-rectangle extent (the opening
    is real, empty, intentional space) so it exercises the same "is this
    reliable" question a naive rectangle-fill check gets wrong."""
    H, W = canvas_hw
    gray = np.full((H, W), bg_value, dtype=np.uint8)
    cv2.ellipse(gray, center, (arc_radius, arc_radius), 0, start_angle, end_angle, fg_value, arm_width)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img, {"center": center, "arc_radius": arc_radius, "arm_width": arm_width}


def make_jagged_noise_blob(
    canvas_hw: tuple[int, int] = (500, 500), center: tuple[int, int] = (250, 250),
    base_radius: int = 120, spike_count: int = 40, spike_variation: int = 90,
    bg_value: int = 30, fg_value: int = 190, seed: int = 3,
) -> tuple[np.ndarray, dict]:
    """A deliberately jagged, star-burst-shaped blob — simulates a genuinely
    noisy/fragmented segmentation (not a legitimate industrial part shape),
    used to confirm the solidity/compactness gates still reject real noise
    even though the bounding-rect-fill gate they replaced is gone."""
    H, W = canvas_hw
    gray = np.full((H, W), bg_value, dtype=np.uint8)
    rng = np.random.RandomState(seed)

    angles = np.linspace(0, 2 * math.pi, spike_count, endpoint=False)
    pts = []
    for a in angles:
        r = base_radius + rng.randint(-spike_variation, spike_variation)
        r = max(10, r)
        pts.append((int(center[0] + r * math.cos(a)), int(center[1] + r * math.sin(a))))
    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(gray, pts, fg_value)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img, {"center": center}


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
