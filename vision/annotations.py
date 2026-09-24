"""Draws the professional-looking overlay: outline, bounding boxes, circles,
dimension text, coordinate axes, and a calibration status banner."""
from __future__ import annotations

import cv2
import numpy as np

from vision.types import CalibrationProfile, CircleFeature, DetectedPart, HoleCandidate, SegmentationDiagnostics

COLOR_CONTOUR = (0, 255, 0)
COLOR_BBOX = (255, 0, 0)
COLOR_ROTATED_RECT = (255, 0, 255)
COLOR_CIRCLE = (0, 165, 255)
COLOR_BORE = (255, 0, 200)
COLOR_CIRCLE_CENTER = (0, 0, 255)
COLOR_TEXT = (0, 255, 255)
COLOR_AXIS_X = (60, 60, 255)
COLOR_AXIS_Y = (60, 220, 60)
COLOR_ACCEPTED = (0, 220, 0)
COLOR_REJECTED = (0, 0, 230)
COLOR_HULL = (0, 220, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fmt(px_value: float | None, mm_value: float | None, unit: str) -> str:
    if mm_value is not None:
        return f"{mm_value:.2f}mm"
    if px_value is not None:
        return f"{px_value:.1f}px"
    return "n/a"


def draw_full_annotation(
    img: np.ndarray,
    part: DetectedPart,
    circles: list[CircleFeature],
    profile: CalibrationProfile | None,
    origin_px: tuple[float, float] | None = None,
) -> np.ndarray:
    out = img.copy()

    cv2.drawContours(out, [part.contour], -1, COLOR_CONTOUR, 2)

    x, y, w, h = part.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_BBOX, 2)

    cx, cy, rw, rh, angle = part.rotated_rect
    box_pts = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(int)
    cv2.drawContours(out, [box_pts], -1, COLOR_ROTATED_RECT, 1)

    w_mm = None if profile is None else w * profile.mm_per_pixel
    h_mm = None if profile is None else h * profile.mm_per_pixel
    label = f"WIDTH = {_fmt(w, w_mm, 'mm')}   HEIGHT = {_fmt(h, h_mm, 'mm')}"
    text_y = max(30, y - 12)
    cv2.putText(out, label, (max(10, x), text_y), FONT, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)

    for c in circles:
        p = (int(round(c.cx)), int(round(c.cy)))
        r = int(round(c.r))
        color = COLOR_BORE if c.feature_type == "bore" else COLOR_CIRCLE
        if c.major_px is not None and c.minor_px is not None and c.ellipse_angle_deg is not None:
            axes = (int(round(c.major_px / 2)), int(round(c.minor_px / 2)))
            cv2.ellipse(out, p, axes, c.ellipse_angle_deg, 0, 360, color, 2)
        else:
            cv2.circle(out, p, r, color, 2)
        cv2.circle(out, p, 3, COLOR_CIRCLE_CENTER, -1)
        d_mm = None if profile is None else 2 * c.r * profile.mm_per_pixel
        short = c.short_id or f"H{c.circle_id}"
        text = f"{short} D={_fmt(2 * c.r, d_mm, 'mm')}"
        cv2.putText(out, text, (max(5, p[0] - r), max(20, p[1] - r - 8)),
                    FONT, 0.55, color, 2, cv2.LINE_AA)

    origin = origin_px or (float(x), float(y))
    axis_len = max(20, int(0.08 * max(w, h)))
    ox, oy = int(origin[0]), int(origin[1])
    cv2.arrowedLine(out, (ox, oy), (ox + axis_len, oy), COLOR_AXIS_X, 2, tipLength=0.25)
    cv2.arrowedLine(out, (ox, oy), (ox, oy - axis_len), COLOR_AXIS_Y, 2, tipLength=0.25)
    cv2.putText(out, "X", (ox + axis_len + 4, oy + 5), FONT, 0.5, COLOR_AXIS_X, 2, cv2.LINE_AA)
    cv2.putText(out, "Y", (ox - 4, oy - axis_len - 6), FONT, 0.5, COLOR_AXIS_Y, 2, cv2.LINE_AA)

    banner = "CALIBRATED" if profile is not None else "NOT CALIBRATED — PIXELS ONLY"
    banner_color = (0, 200, 0) if profile is not None else (0, 0, 220)
    cv2.putText(out, banner, (10, out.shape[0] - 12), FONT, 0.7, banner_color, 2, cv2.LINE_AA)

    return out


def draw_boundary_debug(img: np.ndarray, part: DetectedPart) -> np.ndarray:
    """Vision Debug Mode boundary panel: outer contour, axis-aligned bbox,
    rotated bbox, and convex hull all overlaid, so it's visible exactly how
    much of the "missing" bounding-rectangle area is a genuine concavity
    (inside the hull, outside the contour) versus outside the hull entirely.
    Called even when the boundary was REJECTED — see app.py — so rejection
    reasons are inspectable, not just asserted."""
    out = img.copy()

    cv2.drawContours(out, [part.contour], -1, COLOR_CONTOUR, 2)

    x, y, w, h = part.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_BBOX, 2)

    cx, cy, rw, rh, angle = part.rotated_rect
    box_pts = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(int)
    cv2.drawContours(out, [box_pts], -1, COLOR_ROTATED_RECT, 2)

    hull = cv2.convexHull(part.contour)
    cv2.drawContours(out, [hull], -1, COLOR_HULL, 2)

    legend = [
        ("Contour", COLOR_CONTOUR), ("Bounding box", COLOR_BBOX),
        ("Rotated rect", COLOR_ROTATED_RECT), ("Convex hull", COLOR_HULL),
    ]
    for i, (label, color) in enumerate(legend):
        ly = 24 + i * 22
        cv2.rectangle(out, (10, ly - 12), (26, ly + 2), color, -1)
        cv2.putText(out, label, (32, ly), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_hole_candidates_debug(img: np.ndarray, candidates: list[HoleCandidate]) -> np.ndarray:
    """Vision Debug Mode overlay: every hole candidate that was evaluated,
    green if accepted and red if rejected, so a false-positive flood is
    visible (and explainable) rather than silently hidden."""
    out = img.copy()
    for c in candidates:
        color = COLOR_ACCEPTED if c.accepted else COLOR_REJECTED
        cv2.drawContours(out, [c.contour], -1, color, 1)
        p = (int(round(c.cx)), int(round(c.cy)))
        cv2.circle(out, p, 2, color, -1)
        if c.accepted:
            cv2.putText(out, f"OK {c.confidence_score * 100:.0f}%", (p[0] + 4, p[1] - 4),
                        FONT, 0.4, color, 1, cv2.LINE_AA)
    return out
