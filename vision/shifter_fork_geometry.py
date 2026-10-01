"""Shifter-fork landmark extraction: pixel-space geometry only, no mm
conversion, no recipe/tolerance knowledge (mirrors the existing
`vision.geometry` / `vision.measurement` split). Consumed by
`vision.shifter_fork_measurement`.

Shape model: a two-pronged fork with a U-shaped inner slot (two straight,
roughly-parallel inner edges meeting a semicircular bottom), one or two
outer dome arcs near the top of the part, and an off-axis circular bore.
Reuses `vision.geometry.find_fork_tips` (shape-agnostic dominant-concavity
detector) to locate the slot, and the same RANSAC circle-fit core used
elsewhere in this codebase for the arc fits.

Documented assumptions (flagged here, in the Debug tab, and in the task
report -- pending validation against a real photographed part):

  - Symmetry axis: the part's own long axis (`geometry.rotated_rect`'s
    angle) through the contour's true area centroid (`cv2.moments`), NOT
    the perpendicular bisector of the two inner-slot edges -- the recipes'
    own tolerance bands for `axis_to_left_inner` don't coincide with half
    of `inner_opening`'s tolerance band, so the drawing treats them as
    independently verifiable, which only makes sense if the axis is
    defined independently of the slot edges.
  - "Left" vs "right" (inner edges, outer arcs) is an internal, consistent
    labeling (by transverse sign relative to the axis) for THIS image, not
    verified against true physical left/right from the drawing's point of
    view -- low-risk for the recipes' "confirmed" rows since those are
    symmetric dimensions (either edge should read close to the same
    value on a conforming part), but noted for any `mapping="assumed"`
    left/right-specific row.
  - Every `*_to_bore_x` attribute (all `mapping="assumed"` in every
    recipe -- informational only, never decides accept/reject) is computed
    as a perpendicular distance from the named reference line to the bore
    center. This is a reasonable, documented interpretation, not a
    verified one -- exactly the uncertainty the recipe's own "assumed" tag
    exists to flag.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from vision import geometry
from vision.types import CircleFeature, DetectedPart

# A windowed circle fit must have at least this fraction of its window's
# points as inliers to be trusted -- stricter than geometry's own default
# (0.55) because here the window is deliberately grown from a small,
# likely-pure-curve seed specifically to avoid picking up straight-edge
# contamination; a weak fit at any window size means the arc genuinely
# isn't there, not that a bigger window would help.
_WINDOW_MIN_INLIER_FRACTION = 0.75
_WINDOW_FRACTIONS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0)
_MIN_WINDOW_POINTS = 12


@dataclass
class ShifterForkLandmarks:
    reject_reason: Optional[str] = None

    axis_origin: Optional[tuple[float, float]] = None
    axis_direction: Optional[tuple[float, float]] = None  # unit vector, points AWAY from tip
    axis_angle_deg: Optional[float] = None

    tip_point: Optional[tuple[float, float]] = None
    top_point: Optional[tuple[float, float]] = None

    left_inner_line: Optional[dict] = None   # {"point": (x,y), "direction": (dx,dy)}
    right_inner_line: Optional[dict] = None
    inner_arc: Optional[dict] = None         # {"cx","cy","radius_px",...} from _ransac_circle_fit

    outer_arc_left: Optional[dict] = None
    outer_arc_right: Optional[dict] = None

    bore: Optional[CircleFeature] = None

    outer_width_px: Optional[float] = None
    outer_left_transverse_px: Optional[float] = None
    outer_right_transverse_px: Optional[float] = None


def _fit_line_raw(points: np.ndarray) -> Optional[dict]:
    if len(points) < 2:
        return None
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return {"point": (float(x0), float(y0)), "direction": (float(vx), float(vy))}


_LINE_MODE_ANGLE_TOLERANCE_DEG = 8.0


def _fit_line_segment(points: np.ndarray) -> Optional[dict]:
    """Robust straight-line fit. A tail run out from the inner arc to a
    prong tip often also includes a short, differently-angled segment (a
    blunt tip face, a tip chamfer on a real part) -- fitting the whole
    tail as one ordinary least-squares line biases the direction toward
    that contamination. Instead: compute each point's local direction (via
    a short forward chord), find the dominant angle by histogram mode
    (the true straight edge is normally the largest single-angle cluster,
    since it contributes many consistent-direction points while a curved
    or differently-angled segment spreads across a range of angles), keep
    only points whose local direction is close to that mode, and fit the
    line to just those inliers."""
    n_pts = len(points)
    if n_pts < 8:
        return _fit_line_raw(points)

    step = max(2, n_pts // 15)
    n = n_pts - step
    if n < 4:
        return _fit_line_raw(points)

    dx = points[step:step + n, 0] - points[:n, 0]
    dy = points[step:step + n, 1] - points[:n, 1]
    norm = np.hypot(dx, dy)
    valid = norm > 1e-6
    if valid.sum() < 4:
        return _fit_line_raw(points)

    angles = np.degrees(np.arctan2(dy[valid], dx[valid])) % 180.0  # undirected line angle
    hist, edges = np.histogram(angles, bins=36, range=(0.0, 180.0))
    mode_angle = (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2.0
    diff = np.abs(((angles - mode_angle + 90.0) % 180.0) - 90.0)
    inlier_local_idx = np.where(diff < _LINE_MODE_ANGLE_TOLERANCE_DEG)[0]
    if len(inlier_local_idx) < 4:
        return _fit_line_raw(points)

    valid_idx = np.where(valid)[0]
    chosen = valid_idx[inlier_local_idx]
    point_idx = np.unique(np.concatenate([chosen, chosen + step]))
    point_idx = point_idx[point_idx < n_pts]
    return _fit_line_raw(points[point_idx])


def _point_to_line_distance_px(point: tuple[float, float], line: dict) -> Optional[float]:
    lx, ly = line["point"]
    dx, dy = line["direction"]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return None
    px, py = point[0] - lx, point[1] - ly
    return abs(px * dy - py * dx) / norm


# A true semicircular slot bottom / dome cannot subtend more than 180
# degrees; a small margin above that accommodates rasterization noise near
# the tangent points. `inlier_fraction` alone is NOT a reliable signal that
# a window is contamination-free -- a window that has grown slightly past
# the true tangent points still reports every point as its own "inlier"
# (the fit just quietly biases to accommodate them), so the angular-span
# cap is the real stopping condition; growing the window as far as
# possible WITHIN that cap (rather than stopping at the first passing
# window) measurably improves radius accuracy, confirmed empirically
# against the synthetic fixture across a range of sizes.
_MAX_ARC_ANGULAR_SPAN_DEG = 185.0


def _windowed_circle_fit(run: np.ndarray, anchor_point: tuple[float, float]) -> Optional[dict]:
    """Grows a window of `run` points around the point nearest
    `anchor_point` and RANSAC-circle-fits it, keeping the LARGEST window
    that still respects the semicircle's 180-degree geometric limit --
    isolates a curved segment (slot bottom / dome apex) from straight
    edges sharing the same contour run. Returns the fit dict plus
    `window_lo`/`window_hi` (indices into `run`) so the caller can split
    off the remaining straight-edge tails."""
    n = len(run)
    if n < _MIN_WINDOW_POINTS:
        return None
    anchor_idx = int(np.argmin(np.hypot(run[:, 0] - anchor_point[0], run[:, 1] - anchor_point[1])))

    best = None
    for frac in _WINDOW_FRACTIONS:
        half = max(_MIN_WINDOW_POINTS // 2, int(n * frac / 2))
        lo, hi = max(0, anchor_idx - half), min(n, anchor_idx + half + 1)
        window = run[lo:hi]
        if len(window) < _MIN_WINDOW_POINTS:
            continue
        fit = geometry._ransac_circle_fit(window)
        if fit is None or fit["inlier_fraction"] < _WINDOW_MIN_INLIER_FRACTION:
            continue
        if fit["angular_span_deg"] > _MAX_ARC_ANGULAR_SPAN_DEG:
            break
        fit = dict(fit)
        fit["window_lo"], fit["window_hi"] = lo, hi
        best = fit
    return best


def _axis_from_part(contour: np.ndarray, tip_point: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Independent symmetry-axis estimate: `rotated_rect`'s long-axis angle
    through the contour's true area centroid, oriented to point AWAY from
    `tip_point` (see module docstring for why this is independent of the
    inner-slot edges rather than their bisector)."""
    m = cv2.moments(contour)
    if m["m00"]:
        origin = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    else:
        x, y, w, h = cv2.boundingRect(contour)
        origin = (x + w / 2.0, y + h / 2.0)

    _cx, _cy, _rw, _rh, angle_deg = geometry.rotated_rect(contour)
    rad = math.radians(angle_deg)
    direction = (math.cos(rad), math.sin(rad))

    # orient away from the tip
    to_origin = (origin[0] - tip_point[0], origin[1] - tip_point[1])
    if to_origin[0] * direction[0] + to_origin[1] * direction[1] < 0:
        direction = (-direction[0], -direction[1])

    return origin, direction, angle_deg


def _transverse_sign(point: tuple[float, float], axis_origin: tuple[float, float], axis_direction: tuple[float, float]) -> float:
    """Signed perpendicular offset of `point` from the axis line -- the
    internal "left" (negative) / "right" (positive) convention documented
    in the module docstring."""
    perp = (-axis_direction[1], axis_direction[0])
    dx, dy = point[0] - axis_origin[0], point[1] - axis_origin[1]
    return dx * perp[0] + dy * perp[1]


def extract_shifter_fork_landmarks(part: Optional[DetectedPart], circles: list[CircleFeature]) -> ShifterForkLandmarks:
    lm = ShifterForkLandmarks()

    if part is None:
        lm.reject_reason = "No part detected."
        return lm
    if part.touches_border:
        lm.reject_reason = "Part touches the image border."
        return lm

    tips = geometry.find_fork_tips(part.contour, part.area_px2)
    if tips is None:
        lm.reject_reason = "Fork opening (inner slot) could not be reliably located."
        return lm

    start_idx, end_idx, far_point = tips["start_idx"], tips["end_idx"], tips["far_point"]
    run = geometry._contour_run(part.contour, start_idx, end_idx)

    inner_arc = _windowed_circle_fit(run, far_point)
    if inner_arc is None:
        lm.reject_reason = "Inner arc fit failed."
        return lm
    lm.inner_arc = inner_arc

    # `far_point` (the raw convexity-defect vertex) is only a rough seed --
    # it's a real contour point but not reliably the arc's true apex, since
    # convexity-defect depth is measured against the hull chord, not
    # against the fitted circle. Used here only to orient the axis
    # (direction, not precision); refined below once the axis is known.
    axis_origin, axis_direction, axis_angle_deg = _axis_from_part(part.contour, far_point)
    lm.axis_origin, lm.axis_direction, lm.axis_angle_deg = axis_origin, axis_direction, axis_angle_deg

    # Precise tip_point: the point ON the fitted inner-arc circle in the
    # +axis direction (same construction as the outer apex below) --
    # accurate to the circle fit rather than to a single discretized
    # contour vertex.
    lm.tip_point = (
        inner_arc["cx"] + inner_arc["radius_px"] * axis_direction[0],
        inner_arc["cy"] + inner_arc["radius_px"] * axis_direction[1],
    )

    # Each tail runs from the arc's tangent point out to the prong tip
    # (which may add its own short, differently-angled segment -- see
    # `_fit_line_segment`'s docstring for how that's handled robustly).
    tail_a = run[:inner_arc["window_lo"]]
    tail_b = run[inner_arc["window_hi"]:]
    line_a = _fit_line_segment(tail_a)
    line_b = _fit_line_segment(tail_b)
    if line_a is not None and line_b is not None:
        sign_a = _transverse_sign(line_a["point"], axis_origin, axis_direction)
        sign_b = _transverse_sign(line_b["point"], axis_origin, axis_direction)
        if sign_a <= sign_b:
            lm.left_inner_line, lm.right_inner_line = line_a, line_b
        else:
            lm.left_inner_line, lm.right_inner_line = line_b, line_a

    contour_pts = part.contour.reshape(-1, 2).astype(np.float64)
    projections = (contour_pts[:, 0] - axis_origin[0]) * axis_direction[0] + \
                  (contour_pts[:, 1] - axis_origin[1]) * axis_direction[1]
    # A rounded apex is locally near-flat, so several contour pixels sit
    # within a fraction of a pixel of the true maximum projection -- taking
    # a single argmax picks an arbitrary one of them (whichever comes first
    # in contour order), which can land visibly off-center. Averaging every
    # point within the plateau gives a centered, stable estimate instead.
    max_proj = float(np.max(projections))
    plateau = contour_pts[projections >= max_proj - 1.5]
    lm.top_point = (float(plateau[:, 0].mean()), float(plateau[:, 1].mean()))

    perp = (-axis_direction[1], axis_direction[0])
    transverse = (contour_pts[:, 0] - axis_origin[0]) * perp[0] + (contour_pts[:, 1] - axis_origin[1]) * perp[1]
    lm.outer_left_transverse_px = float(np.min(transverse))
    lm.outer_right_transverse_px = float(np.max(transverse))
    lm.outer_width_px = lm.outer_right_transverse_px - lm.outer_left_transverse_px

    comp_run = geometry._contour_run(part.contour, end_idx, start_idx)
    dome_fit = _windowed_circle_fit(comp_run, lm.top_point)
    if dome_fit is None:
        lm.reject_reason = "Outer arc fit failed."
        return lm

    dome_window = comp_run[dome_fit["window_lo"]:dome_fit["window_hi"]]
    dome_signs = np.array([_transverse_sign((p[0], p[1]), axis_origin, axis_direction) for p in dome_window])
    left_pts = dome_window[dome_signs <= 0]
    right_pts = dome_window[dome_signs > 0]

    left_fit = geometry._ransac_circle_fit(left_pts) if len(left_pts) >= _MIN_WINDOW_POINTS else None
    right_fit = geometry._ransac_circle_fit(right_pts) if len(right_pts) >= _MIN_WINDOW_POINTS else None
    lm.outer_arc_left = left_fit if left_fit is not None else dome_fit
    lm.outer_arc_right = right_fit if right_fit is not None else dome_fit

    bore = next((c for c in circles if c.feature_type == "bore"), None) or (
        max(circles, key=lambda c: c.r) if circles else None
    )
    if bore is None:
        lm.reject_reason = "Bore not detected."
        return lm
    lm.bore = bore

    return lm
