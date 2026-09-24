"""Pixel-space geometry: bounding boxes, circles/holes, distances, corner
radii, and angles. Everything here operates in pixels — `vision.measurement`
is responsible for converting to mm using a calibration profile.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from vision.types import CircleFeature, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW

# --- circle detection thresholds -------------------------------------------------
# A contour-based hole candidate must be at least this circular
# (4*pi*Area/Perimeter^2, 1.0 = perfect circle) to be trusted at all.
MIN_CIRCULARITY_TO_ACCEPT = 0.70
MIN_CIRCULARITY_FOR_HIGH_CONFIDENCE = 0.90
# Hough/contour detections whose centers are within this many pixels are
# treated as the same physical circle.
DEDUP_DISTANCE_PX = 12.0

# --- corner radius fit thresholds -------------------------------------------------
MIN_CORNER_FIT_POINTS = 6
MAX_CORNER_RESIDUAL_RATIO = 0.15  # RMS fit error must be < 15% of fitted radius
MIN_CORNER_EDGE_LEN_PX = 6.0


def bounding_box(contour: np.ndarray) -> tuple[int, int, int, int]:
    return cv2.boundingRect(contour)


def rotated_rect(contour: np.ndarray) -> tuple[float, float, float, float, float]:
    """Returns (cx, cy, width, height, angle_degrees) with width >= height,
    angle normalized to [-90, 90)."""
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    angle = ((angle + 90.0) % 180.0) - 90.0
    return cx, cy, rw, rh, angle


def area_perimeter(contour: np.ndarray) -> tuple[float, float]:
    return float(cv2.contourArea(contour)), float(cv2.arcLength(contour, True))


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def distance_to_contour(contour: np.ndarray, point: tuple[float, float]) -> float:
    """Absolute distance in px from `point` to the nearest point on `contour`."""
    return abs(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))


# --- circle / hole detection -------------------------------------------------------

def detect_circles_hough(gray: np.ndarray, mask: np.ndarray | None) -> list[dict]:
    blurred = cv2.medianBlur(gray, 5)
    H, W = gray.shape
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(H, W) // 12),
        param1=100,
        param2=30,
        minRadius=max(5, min(H, W) // 100),
        maxRadius=max(10, min(H, W) // 3),
    )
    results = []
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            if not (0 <= x < W and 0 <= y < H):
                continue
            if mask is not None and mask[y, x] == 0:
                continue
            if x - r < 0 or y - r < 0 or x + r >= W or y + r >= H:
                continue
            results.append({"cx": float(x), "cy": float(y), "r": float(r), "method": "hough", "circularity": None})
    return results


def detect_circles_contour(hole_contours: list[np.ndarray]) -> list[dict]:
    results = []
    for c in hole_contours:
        area, perimeter = area_perimeter(c)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY_TO_ACCEPT:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        results.append({
            "cx": float(cx), "cy": float(cy), "r": float(r),
            "method": "contour", "circularity": float(circularity),
        })
    return results


def merge_circle_detections(hough: list[dict], contour: list[dict]) -> list[CircleFeature]:
    """Deduplicate by center proximity. A circle confirmed by both methods, or
    by contour circularity alone at high circularity, gets HIGH confidence.
    A Hough-only hit (no circularity evidence) is capped at MEDIUM."""
    merged: list[dict] = []
    used_contour = [False] * len(contour)

    for h in hough:
        match_idx = None
        for i, c in enumerate(contour):
            if used_contour[i]:
                continue
            if euclidean_distance((h["cx"], h["cy"]), (c["cx"], c["cy"])) <= DEDUP_DISTANCE_PX:
                match_idx = i
                break
        if match_idx is not None:
            c = contour[match_idx]
            used_contour[match_idx] = True
            circularity = c["circularity"]
            confidence = CONFIDENCE_HIGH if circularity >= MIN_CIRCULARITY_FOR_HIGH_CONFIDENCE else CONFIDENCE_MEDIUM
            # Prefer the contour-based radius/center (measured from the hole
            # boundary directly) over the Hough estimate.
            merged.append({"cx": c["cx"], "cy": c["cy"], "r": c["r"], "method": "hough+contour",
                            "circularity": circularity, "confidence": confidence})
        else:
            merged.append({"cx": h["cx"], "cy": h["cy"], "r": h["r"], "method": "hough",
                            "circularity": None, "confidence": CONFIDENCE_MEDIUM})

    for i, c in enumerate(contour):
        if used_contour[i]:
            continue
        confidence = CONFIDENCE_HIGH if c["circularity"] >= MIN_CIRCULARITY_FOR_HIGH_CONFIDENCE else CONFIDENCE_MEDIUM
        merged.append({"cx": c["cx"], "cy": c["cy"], "r": c["r"], "method": "contour",
                        "circularity": c["circularity"], "confidence": confidence})

    features = []
    for i, m in enumerate(merged, start=1):
        features.append(CircleFeature(
            circle_id=i, cx=m["cx"], cy=m["cy"], r=m["r"],
            method=m["method"], circularity=m["circularity"], confidence=m["confidence"],
        ))
    return features


# --- corner radius + angle estimation ----------------------------------------------

def polygon_approx(contour: np.ndarray, epsilon_ratio: float = 0.01) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.reshape(-1, 2)


def _fit_circle_least_squares(points: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) least-squares circle fit. Returns (cx, cy, r, rms_error)."""
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r_sq = sol[2] + cx ** 2 + cy ** 2
    if r_sq <= 0:
        return cx, cy, 0.0, float("inf")
    r = math.sqrt(r_sq)
    dists = np.hypot(x - cx, y - cy)
    rms = float(np.sqrt(np.mean((dists - r) ** 2)))
    return cx, cy, r, rms


def estimate_corner_radii(contour: np.ndarray, epsilon_ratio: float = 0.02) -> list[dict]:
    """For each simplified-polygon vertex, fit a circle to the nearby raw
    contour points and report a radius only if the fit is tight. Otherwise
    the corner is reported as unreliable (never a guessed number)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 8:
        return []

    vertices = polygon_approx(contour, epsilon_ratio)
    if len(vertices) < 3:
        return []

    part_bbox_w, part_bbox_h = cv2.boundingRect(contour)[2:]
    max_absolute_r = 0.3 * min(part_bbox_w, part_bbox_h)

    def _window_points(center_idx: int, half_window: int) -> np.ndarray:
        half_window = max(MIN_CORNER_FIT_POINTS, min(half_window, n // 2))
        lo = (center_idx - half_window) % n
        return np.take(pts, [(lo + k) % n for k in range(2 * half_window)], axis=0)

    # Map each simplified vertex to its nearest index in the raw contour.
    results = []
    for i, v in enumerate(vertices):
        idx = int(np.argmin(np.sum((pts - v) ** 2, axis=1)))
        prev_v = vertices[i - 1]
        next_v = vertices[(i + 1) % len(vertices)]
        edge_prev_len = euclidean_distance(tuple(v), tuple(prev_v))
        edge_next_len = euclidean_distance(tuple(v), tuple(next_v))

        entry = {"x": float(v[0]), "y": float(v[1]), "radius_px": None, "reliable": False, "reason": ""}

        if edge_prev_len < MIN_CORNER_EDGE_LEN_PX or edge_next_len < MIN_CORNER_EDGE_LEN_PX:
            entry["reason"] = "Adjacent edges too short to fit reliably"
            results.append(entry)
            continue

        # Pass 1: a small seed window strictly local to the vertex — using
        # the full adjacent-edge length here (as a first cut of this code
        # did) lets the window bleed into the straight edges on either side
        # of the arc, which biases a least-squares circle fit toward a much
        # larger radius (straight points look like an arc of infinite R).
        seed_half_window = int(np.clip(min(edge_prev_len, edge_next_len) * 0.12, MIN_CORNER_FIT_POINTS, 20))
        window_pts = _window_points(idx, seed_half_window)
        if len(window_pts) < MIN_CORNER_FIT_POINTS:
            entry["reason"] = "Not enough contour points near this corner"
            results.append(entry)
            continue

        cx, cy, r, rms = _fit_circle_least_squares(window_pts)

        # Pass 2: resize the window to the arc length implied by the seed
        # estimate (a ~90-degree turn spans about r*pi/2 contour points at
        # ~1px spacing) and refit once for a tighter, less-contaminated fit.
        if r > 0:
            refined_half_window = int(np.clip(r * math.pi / 2 * 0.6, MIN_CORNER_FIT_POINTS, 60))
            refined_pts = _window_points(idx, refined_half_window)
            if len(refined_pts) >= MIN_CORNER_FIT_POINTS:
                cx, cy, r, rms = _fit_circle_least_squares(refined_pts)
                window_pts = refined_pts

        max_reasonable_r = min(0.5 * min(edge_prev_len, edge_next_len) + 5, max_absolute_r)
        if r <= 0 or r > max_reasonable_r or rms > MAX_CORNER_RESIDUAL_RATIO * max(r, 1e-6):
            entry["reason"] = "Radius could not be reliably determined"
            results.append(entry)
            continue

        entry["radius_px"] = float(r)
        entry["reliable"] = True
        entry["fit_rms_px"] = float(rms)
        results.append(entry)

    return results


def estimate_angles(contour: np.ndarray, epsilon_ratio: float = 0.02) -> list[dict]:
    """Interior angles (degrees) at each stable simplified-polygon vertex."""
    vertices = polygon_approx(contour, epsilon_ratio)
    n = len(vertices)
    if n < 3:
        return []

    results = []
    for i, v in enumerate(vertices):
        prev_v = vertices[i - 1]
        next_v = vertices[(i + 1) % n]
        v1 = prev_v - v
        v2 = next_v - v
        len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)

        entry = {"x": float(v[0]), "y": float(v[1]), "angle_deg": None, "reliable": False}
        if len1 < MIN_CORNER_EDGE_LEN_PX or len2 < MIN_CORNER_EDGE_LEN_PX:
            results.append(entry)
            continue

        cos_theta = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
        angle = math.degrees(math.acos(cos_theta))
        entry["angle_deg"] = float(angle)
        entry["reliable"] = True
        results.append(entry)

    return results
