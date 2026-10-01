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


def make_fork_bracket_part(
    canvas_hw: tuple[int, int] = (900, 1400), hub_center: tuple[int, int] = (700, 550),
    inner_radius: int = 180, outer_radius: int = 250,
    start_angle: float = 200, end_angle: float = 340,
    main_bore_r: int = 30, boss_hole_r: int = 10, boss_hole_offset: tuple[int, int] = (-35, 35),
    bg_value: int = 25, fg_value: int = 195,
) -> tuple[np.ndarray, dict]:
    """A fork/rocker-arm-style bracket with known ground truth for every new
    fork-geometry function: two arms of constant radial thickness
    (`outer_radius - inner_radius`) sweeping from `start_angle` to
    `end_angle` around `hub_center` (the arc's pivot point, NOT necessarily
    filled material itself), each ending in a straight RADIAL tip face —
    built by explicitly tracing outer arc -> radial tip face -> inner arc
    -> matching radial tip face as one filled polygon, so the tip faces are
    geometrically exact (unlike `make_c_shape_part`, whose `cv2.ellipse`
    thickness rendering doesn't guarantee a clean straight cut at the
    ends). A hub "boss" bump is added near the sweep's midpoint, centered
    just inside the inner boundary and sized so it overlaps the arm band
    without ever reaching `outer_radius` (so it can't distort the outer
    convex hull the tip-finder relies on) while still intruding well past
    `inner_radius` — a local, non-circular outlier in the inner-arc region,
    like the real part's hub, so `fit_concave_arc`'s outlier-robustness is
    actually exercised. The main bore and boss hole are cut into the hub
    bump itself (real filled material), not the arc's abstract pivot
    point, matching where the real part's holes actually sit."""
    H, W = canvas_hw
    gray = np.full((H, W), bg_value, dtype=np.uint8)
    cx, cy = hub_center

    n_arc_pts = 120
    angles = np.linspace(math.radians(start_angle), math.radians(end_angle), n_arc_pts)
    outer_pts = [(cx + outer_radius * math.cos(a), cy + outer_radius * math.sin(a)) for a in angles]
    inner_pts = [(cx + inner_radius * math.cos(a), cy + inner_radius * math.sin(a)) for a in reversed(angles)]
    poly = np.array(outer_pts + inner_pts, dtype=np.int32)
    cv2.fillPoly(gray, [poly], fg_value)

    mid_angle = math.radians((start_angle + end_angle) / 2)
    hub_dist = inner_radius - 20  # just inside the inner boundary
    hub_r = 70                    # reach: [hub_dist - hub_r, hub_dist + hub_r] = [90, 230] -- overlaps
                                   # the arm band (180-250) without ever reaching outer_radius=250,
                                   # and big enough relative to main_bore_r/boss_hole_r that both
                                   # holes fit inside it without touching each other or its edge
    hub_center_xy = (int(cx + hub_dist * math.cos(mid_angle)), int(cy + hub_dist * math.sin(mid_angle)))
    cv2.circle(gray, hub_center_xy, hub_r, fg_value, -1)

    cv2.circle(gray, hub_center_xy, main_bore_r, bg_value, -1)
    boss_center = (hub_center_xy[0] + boss_hole_offset[0], hub_center_xy[1] + boss_hole_offset[1])
    cv2.circle(gray, boss_center, boss_hole_r, bg_value, -1)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    tip_a_outer, tip_b_outer = outer_pts[0], outer_pts[-1]
    tip_a_inner, tip_b_inner = inner_pts[-1], inner_pts[0]
    tip_gap_outer_px = math.hypot(tip_a_outer[0] - tip_b_outer[0], tip_a_outer[1] - tip_b_outer[1])

    return img, {
        "hub_center": hub_center, "inner_radius": inner_radius, "outer_radius": outer_radius,
        "arm_thickness_px": outer_radius - inner_radius,
        "tip_a_outer": tip_a_outer, "tip_b_outer": tip_b_outer,
        "tip_a_inner": tip_a_inner, "tip_b_inner": tip_b_inner,
        "tip_gap_outer_px": tip_gap_outer_px,
        "main_bore": (hub_center_xy[0], hub_center_xy[1], main_bore_r),
        "boss_hole": (boss_center[0], boss_center[1], boss_hole_r),
    }


def make_shifter_fork_part(
    inner_opening_px: float = 200.0,
    straight_side_len_px: float = 60.0,
    prong_wall_px: float = 25.0,
    boss_straight_height_px: float = 35.0,
    bore_radius_px: float = 18.0,
    bore_transverse_offset_px: float = -35.0,
    bg_value: int = 225, fg_value: int = 40,
    margin: int = 80,
) -> tuple[np.ndarray, dict]:
    """A two-pronged shifter-fork silhouette: two straight parallel inner
    edges meeting a semicircular slot bottom (radius = inner_opening/2, as
    in the real recipes where "Fork Half Opening" nominal equals exactly
    half of "Inner Fork Opening"), straight prong outer walls, a
    semicircular outer dome directly above (radius = outer half-width, so
    it meets the straight walls tangentially -- the dome apex is this
    fixture's `top_point`, i.e. `tip_to_top` == `tip_to_outer_apex` here;
    a real part's extra boss material above the dome is not modeled), and
    an off-axis bore drilled into the boss above the slot.

    Backlit-silhouette polarity, per the real recipe's brief: part dark
    (`fg_value`), background bright (`bg_value`), bore bright (same as
    background -- a real hole, not a coincidence of the fill values)."""
    half_opening = inner_opening_px / 2.0
    arc_radius = half_opening
    outer_half_width = arc_radius + prong_wall_px
    shoulder_y = arc_radius + boss_straight_height_px  # part-frame y where the dome starts
    apex_y = shoulder_y + outer_half_width              # dome apex = top_point

    n_arc = 60
    left_outer_bottom = (-outer_half_width, -straight_side_len_px)
    left_outer_top = (-outer_half_width, shoulder_y)
    dome_pts = [(outer_half_width * math.cos(t), shoulder_y + outer_half_width * math.sin(t))
                for t in np.linspace(math.pi, 0.0, n_arc)]
    right_outer_bottom = (outer_half_width, -straight_side_len_px)
    right_inner_bottom = (arc_radius, -straight_side_len_px)
    slot_pts = [(arc_radius * math.cos(t), arc_radius * math.sin(t))
                for t in np.linspace(0.0, math.pi, n_arc)]
    left_inner_bottom = (-arc_radius, -straight_side_len_px)

    part_pts = [left_outer_bottom, left_outer_top] + dome_pts + \
               [right_outer_bottom, right_inner_bottom] + slot_pts + [left_inner_bottom]

    def to_image(pt):
        return (margin + outer_half_width + pt[0], margin + apex_y - pt[1])

    img_pts = np.array([to_image(p) for p in part_pts], dtype=np.int32)

    W = int(round(2 * margin + 2 * outer_half_width))
    H = int(round(2 * margin + apex_y + straight_side_len_px))
    gray = np.full((H, W), bg_value, dtype=np.uint8)
    cv2.fillPoly(gray, [img_pts], fg_value)

    bore_y_pf = arc_radius + boss_straight_height_px * 0.5
    bore_cx, bore_cy = to_image((bore_transverse_offset_px, bore_y_pf))
    if bore_radius_px > 0:
        cv2.circle(gray, (int(round(bore_cx)), int(round(bore_cy))), int(round(bore_radius_px)), bg_value, -1)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    tip_point = to_image((0.0, arc_radius))
    top_point = to_image((0.0, apex_y))

    return img, {
        "inner_opening_px": inner_opening_px,
        "inner_arc_radius_px": arc_radius,
        "outer_arc_radius_px": outer_half_width,
        "outer_width_px": 2 * outer_half_width,
        "tip_point": tip_point,
        "top_point": top_point,
        "bore_center": (bore_cx, bore_cy),
        "bore_radius_px": bore_radius_px,
        "axis_direction": (0.0, -1.0),  # "away from tip" points up the image
    }


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
