"""Pixel-space geometry: bounding boxes, circles/holes, distances, corner
radii, and angles. Everything here operates in pixels — `vision.measurement`
is responsible for converting to mm using a calibration profile.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from vision.types import (
    CircleFeature, HoleCandidate, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW,
)

# --- hole acceptance gates ---------------------------------------------------------
# A hole candidate must clear EVERY one of these before it is even considered
# for acceptance (see `evaluate_hole_candidates`). None of these alone is
# sufficient — text glyphs, logos, texture speckle, and reflections can each
# pass one or two checks individually, which is why a real product photo
# needs all of them together to avoid mass false-positive floods.
MIN_CIRCULARITY = 0.72          # 4*pi*Area/Perimeter^2; 1.0 = perfect circle
MIN_SOLIDITY = 0.85             # Area / convex-hull area; rejects strokes, irregular blobs
MAX_ASPECT_RATIO = 2.3          # ellipse major/minor; allows perspective skew, not scratches
MIN_CONTRAST = 14.0             # |interior mean - surrounding ring mean|, 0-255 gray levels
MAX_INTERIOR_STD = 22.0         # std dev WITHIN the interior; high = not a consistent surface
                                 # (catches ring/stroke shapes like printed "O"/"0"/"Q"/"8" text,
                                 # whose outer boundary alone looks deceptively hole-like — see
                                 # `_local_contrast_and_uniformity`'s docstring)
MIN_EDGE_STRENGTH = 18.0        # mean Sobel gradient magnitude along the boundary
MIN_DIAMETER_PX = 6.0           # anything smaller is unmeasurable/noise-prone regardless of image size
MIN_CONFIDENCE_SCORE_TO_ACCEPT = 0.55  # 0-1 aggregate score, see `_confidence_score`

# Points/circle centers within this many pixels are treated as the same
# physical feature (both for Hough<->contour cross-checking and final dedup).
DEDUP_DISTANCE_PX = 12.0

# A contour-based hole candidate must be at least this circular
# (4*pi*Area/Perimeter^2, 1.0 = perfect circle) to be trusted at all.
MIN_CIRCULARITY_TO_ACCEPT = MIN_CIRCULARITY  # kept for backwards-compat readability
MIN_CIRCULARITY_FOR_HIGH_CONFIDENCE = 0.90

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
#
# Design note (why this isn't "just call HoughCircles"): a candidate is never
# reported as a hole on the strength of a single signal. HoughCircles alone
# false-positives heavily on text, logos, reflections, and texture in real
# photos — so it is used ONLY as a cross-check that can raise an already-
# contour-confirmed candidate's confidence, never as a standalone source of
# truth. Every actual candidate instead comes from `segmentation.
# extract_hole_contours` (an enclosed region inside the part), and must then
# clear circularity, solidity, aspect-ratio, contrast, edge-strength, and
# size gates — all evaluated here — before it can be accepted at all.


def detect_circles_hough(
    gray: np.ndarray, mask: np.ndarray | None, bbox: tuple[int, int, int, int] | None = None,
) -> list[dict]:
    """Raw Hough-circle hits, scaled to the detected PART's size (not the
    whole image) when `bbox` is given — a fixed image-relative scale is
    poorly calibrated when the part only fills part of the frame. Returned
    hits are only ever used as a cross-check signal; see module docstring."""
    blurred = cv2.medianBlur(gray, 5)
    H, W = gray.shape
    if bbox is not None:
        _, _, bw, bh = bbox
        part_scale = max(10, min(bw, bh))
    else:
        part_scale = min(H, W)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(15, part_scale // 15),
        param1=100,
        param2=28,
        minRadius=max(4, part_scale // 120),
        maxRadius=max(15, part_scale // 4),
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
            results.append({"cx": float(x), "cy": float(y), "r": float(r)})
    return results


def _solidity(contour: np.ndarray, area: float) -> float:
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


def _fit_ellipse_safe(contour: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """Returns (cx, cy, major_axis, minor_axis, angle_deg) or None. `major`
    and `minor` are full axis lengths (diameters), major >= minor."""
    if len(contour) < 5:
        return None
    try:
        (ecx, ecy), (d1, d2), angle = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    major, minor = max(d1, d2), min(d1, d2)
    if minor <= 0:
        return None
    return float(ecx), float(ecy), float(major), float(minor), float(angle)


def _local_contrast_and_uniformity(
    gray: np.ndarray, contour: np.ndarray, outer_mask: np.ndarray,
) -> tuple[float, float]:
    """Returns (contrast, interior_std).

    `contrast` = |mean interior intensity - mean intensity of a thin ring
    just outside the candidate, still inside the part|. A real hole reads as
    a consistently different region from its surrounding material; faint
    JPEG noise or subtle texture does not.

    `interior_std` is the standard deviation of gray values WITHIN the
    filled interior. This catches a failure mode the contrast check alone
    misses: `cv2.contourArea`/`drawContours(..., FILLED)` treat a shape's
    OUTER boundary as if it enclosed a solid region — but a thin ring (e.g. a
    printed letter like "O", "0", "Q", "8", whose ink is a stroke, not a
    fill) has an outer boundary that looks exactly like a small solid hole's
    boundary. The filled interior of a real hole is one consistent surface
    (low std); the "filled interior" of a ring is actually a mix of ink and
    the surrounding material peeking through its hollow center (high std).
    This is the "consistent interior region" a real hole has, per spec.
    """
    x, y, w, h = cv2.boundingRect(contour)
    H, W = gray.shape
    pad = max(4, int(0.6 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

    local_gray = gray[y0:y1, x0:x1]
    local_outer = outer_mask[y0:y1, x0:x1]
    interior_mask = np.zeros(local_gray.shape, dtype=np.uint8)
    shifted = contour - np.array([[x0, y0]])
    cv2.drawContours(interior_mask, [shifted], -1, 255, thickness=cv2.FILLED)

    dilated = cv2.dilate(interior_mask, np.ones((9, 9), np.uint8))
    ring_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(interior_mask))
    ring_mask = cv2.bitwise_and(ring_mask, local_outer)

    if cv2.countNonZero(interior_mask) == 0 or cv2.countNonZero(ring_mask) == 0:
        return 0.0, 255.0

    interior_vals = local_gray[interior_mask == 255].astype(np.float64)
    interior_mean = float(np.mean(interior_vals))
    interior_std = float(np.std(interior_vals))
    ring_mean = cv2.mean(local_gray, mask=ring_mask)[0]
    contrast = abs(interior_mean - ring_mean)
    return contrast, interior_std


def _boundary_edge_strength(grad_mag: np.ndarray, contour: np.ndarray) -> float:
    """Mean gradient magnitude sampled at the candidate's boundary pixels —
    a real hole has a sharp, well-defined edge; a shadow or reflection tends
    to have a soft, low-gradient one."""
    H, W = grad_mag.shape
    pts = contour.reshape(-1, 2)
    xs = np.clip(pts[:, 0], 0, W - 1)
    ys = np.clip(pts[:, 1], 0, H - 1)
    if len(xs) == 0:
        return 0.0
    return float(np.mean(grad_mag[ys, xs]))


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _margin_score(value: float, min_ok: float, good_at: float) -> float:
    """0 at `min_ok`, 1 at `good_at` (or beyond) — how comfortably a metric
    clears its minimum gate, used for the continuous confidence score."""
    if good_at == min_ok:
        return 1.0 if value >= min_ok else 0.0
    return _clamp01((value - min_ok) / (good_at - min_ok))


def _confidence_score(circularity: float, solidity: float, aspect_ratio: float,
                       contrast: float, interior_std: float, edge_strength: float,
                       hough_confirmed: bool) -> float:
    circularity_score = _margin_score(circularity, MIN_CIRCULARITY, 0.95)
    solidity_score = _margin_score(solidity, MIN_SOLIDITY, 0.98)
    contrast_score = _margin_score(contrast, MIN_CONTRAST, 45.0)
    # interior_std is a "lower is better" metric, so the margin runs the
    # other way: 1.0 at 0 std, 0.0 at/beyond MAX_INTERIOR_STD.
    uniformity_score = _margin_score(MAX_INTERIOR_STD - interior_std, 0.0, MAX_INTERIOR_STD)
    edge_score = _margin_score(edge_strength, MIN_EDGE_STRENGTH, 45.0)
    aspect_penalty = _clamp01((aspect_ratio - 1.0) / max(MAX_ASPECT_RATIO - 1.0, 1e-6))
    aspect_score = 1.0 - aspect_penalty

    base = float(np.mean([circularity_score, solidity_score, contrast_score, uniformity_score, edge_score, aspect_score]))
    bonus = 0.10 if hough_confirmed else 0.0
    return min(1.0, base + bonus)


def evaluate_hole_candidates(
    gray: np.ndarray,
    outer_mask: np.ndarray,
    hole_contours: list[np.ndarray],
    hough_hits: list[dict],
    part_area_px2: float,
) -> list[HoleCandidate]:
    """The actual accept/reject decision. Every raw candidate from
    `segmentation.extract_hole_contours` is evaluated against every gate and
    kept in the returned list REGARDLESS of outcome (with `.accepted` and
    `.rejection_reasons` set) — Vision Debug Mode uses the full list; normal
    measurement uses only the accepted ones via
    `hole_candidates_to_circle_features`."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    hough_used = [False] * len(hough_hits)
    results: list[HoleCandidate] = []

    for c in hole_contours:
        area, perimeter = area_perimeter(c)
        if perimeter <= 0 or area <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        solidity = _solidity(c, area)
        equiv_diameter = 2.0 * math.sqrt(area / math.pi)

        ellipse = _fit_ellipse_safe(c)
        if ellipse is not None:
            ecx, ecy, major, minor, angle = ellipse
            aspect_ratio = major / minor
        else:
            (ecx, ecy), r = cv2.minEnclosingCircle(c)
            major = minor = 2.0 * r
            angle = 0.0
            aspect_ratio = 1.0

        M = cv2.moments(c)
        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            cx, cy = ecx, ecy

        contrast, interior_std = _local_contrast_and_uniformity(gray, c, outer_mask)
        edge_strength = _boundary_edge_strength(grad_mag, c)

        hough_confirmed = False
        for i, h in enumerate(hough_hits):
            if hough_used[i]:
                continue
            if euclidean_distance((cx, cy), (h["cx"], h["cy"])) <= DEDUP_DISTANCE_PX:
                # Only count it as confirmation if the radii are also roughly
                # consistent — a nearby Hough hit of a wildly different size
                # is more likely an unrelated coincidence than the same hole.
                if 0.4 <= (h["r"] / max(equiv_diameter / 2.0, 1e-6)) <= 2.5:
                    hough_confirmed = True
                    hough_used[i] = True
                    break

        reasons: list[str] = []
        if circularity < MIN_CIRCULARITY:
            reasons.append(f"Low circularity ({circularity:.2f} < {MIN_CIRCULARITY:.2f})")
        if solidity < MIN_SOLIDITY:
            reasons.append(f"Low solidity / irregular shape ({solidity:.2f} < {MIN_SOLIDITY:.2f})")
        if aspect_ratio > MAX_ASPECT_RATIO:
            reasons.append(f"Too elongated (aspect ratio {aspect_ratio:.2f} > {MAX_ASPECT_RATIO:.2f})")
        if contrast < MIN_CONTRAST:
            reasons.append(f"Weak contrast vs surrounding material ({contrast:.1f} < {MIN_CONTRAST:.1f})")
        if interior_std > MAX_INTERIOR_STD:
            reasons.append(f"Interior is not a consistent surface (std {interior_std:.1f} > {MAX_INTERIOR_STD:.1f}) — "
                            f"looks like a stroke/ring (e.g. text) rather than a filled hole")
        if edge_strength < MIN_EDGE_STRENGTH:
            reasons.append("Weak/soft boundary edges")
        if equiv_diameter < MIN_DIAMETER_PX:
            reasons.append(f"Too small to measure reliably (<{MIN_DIAMETER_PX:.0f}px)")

        score = _confidence_score(circularity, solidity, aspect_ratio, contrast, interior_std, edge_strength, hough_confirmed)
        if not reasons and score < MIN_CONFIDENCE_SCORE_TO_ACCEPT:
            reasons.append(f"Low overall confidence ({score * 100:.0f}%)")

        label = CONFIDENCE_HIGH if score >= 0.80 else (CONFIDENCE_MEDIUM if score >= 0.55 else CONFIDENCE_LOW)

        results.append(HoleCandidate(
            contour=c, cx=float(cx), cy=float(cy), area_px2=float(area),
            equiv_diameter_px=float(equiv_diameter), circularity=float(circularity),
            solidity=float(solidity), aspect_ratio=float(aspect_ratio),
            major_axis_px=float(major) if ellipse is not None else None,
            minor_axis_px=float(minor) if ellipse is not None else None,
            ellipse_angle_deg=float(angle) if ellipse is not None else None,
            contrast=float(contrast), interior_std=float(interior_std), edge_strength_px=float(edge_strength),
            hough_confirmed=hough_confirmed, confidence_score=score, confidence_label=label,
            accepted=not reasons, rejection_reasons=reasons,
        ))

    return _dedup_candidates(results)


def _dedup_candidates(candidates: list[HoleCandidate]) -> list[HoleCandidate]:
    """Defensive final dedup among ACCEPTED candidates by center proximity —
    the two Otsu partitions that generate candidates are complementary so
    this shouldn't normally trigger, but guards against any future extra
    candidate source double-counting the same physical hole. Keeps the
    higher-confidence one and marks the other rejected as a duplicate."""
    accepted_idx = [i for i, c in enumerate(candidates) if c.accepted]
    accepted_idx.sort(key=lambda i: candidates[i].confidence_score, reverse=True)

    kept: list[int] = []
    for i in accepted_idx:
        c = candidates[i]
        is_dup = False
        for j in kept:
            k = candidates[j]
            if euclidean_distance((c.cx, c.cy), (k.cx, k.cy)) <= DEDUP_DISTANCE_PX:
                is_dup = True
                break
        if is_dup:
            c.accepted = False
            c.rejection_reasons.append("Duplicate of another detected hole")
        else:
            kept.append(i)

    return candidates


def detect_holes(
    gray: np.ndarray,
    outer_mask: np.ndarray,
    hole_contours: list[np.ndarray],
    bbox: tuple[int, int, int, int],
    part_area_px2: float,
) -> tuple[list[CircleFeature], list[HoleCandidate]]:
    """Convenience wrapper tying together Hough cross-checking, full
    candidate evaluation, and conversion to the final reportable list.
    Returns (accepted_circles, all_candidates) — the second is what Vision
    Debug Mode uses to show rejected candidates and why."""
    hough_hits = detect_circles_hough(gray, outer_mask, bbox)
    candidates = evaluate_hole_candidates(gray, outer_mask, hole_contours, hough_hits, part_area_px2)
    circles = hole_candidates_to_circle_features(candidates)
    return circles, candidates


def hole_candidates_to_circle_features(candidates: list[HoleCandidate]) -> list[CircleFeature]:
    """The final, reportable result: only candidates that passed every gate,
    converted to the `CircleFeature` shape the rest of the app consumes."""
    features = []
    for i, c in enumerate((x for x in candidates if x.accepted), start=1):
        features.append(CircleFeature(
            circle_id=i, cx=c.cx, cy=c.cy, r=c.equiv_diameter_px / 2.0,
            method="hough+contour" if c.hough_confirmed else "contour",
            circularity=c.circularity, confidence=c.confidence_label,
            major_px=c.major_axis_px, minor_px=c.minor_axis_px,
            ellipse_angle_deg=c.ellipse_angle_deg, solidity=c.solidity,
            confidence_score=c.confidence_score, hough_confirmed=c.hough_confirmed,
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
