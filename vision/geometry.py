"""Pixel-space geometry: bounding boxes, circles/holes, distances, corner
radii, and angles. Everything here operates in pixels — `vision.measurement`
is responsible for converting to mm using a calibration profile.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from vision.types import (
    CircleFeature, DetectedPart, HoleCandidate, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW,
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
MIN_DARK_MAJORITY_FRACTION = 0.25  # if the interior is at least this dark-majority, a high
                                    # interior_std is treated as "real cavity + a highlight",
                                    # not "hollow ring" — see `_local_contrast_and_uniformity`.
                                    # Calibrated against real data: every synthetic text-glyph
                                    # artifact tested scores exactly 0.00 (ink is a thin minority
                                    # stroke), while a real photographed bore with a strong internal
                                    # specular highlight scored 0.35 — this sits with margin on
                                    # both sides of that gap, not centered arbitrarily.
MIN_EDGE_STRENGTH = 18.0        # mean Sobel gradient magnitude along the boundary
MIN_DIAMETER_PX = 12.0          # anything smaller is unmeasurable/noise-prone regardless of image size —
                                 # also the safety margin above the ~8px inner gap of a letter like "O"/
                                 # "Q"/"8" in printed text at ordinary scale, confirmed via a synthetic
                                 # test: recovering genuinely nested hole candidates (see
                                 # `segmentation._find_contours_excluding_hole_traces`) also recovers a
                                 # text glyph's own tiny inner gap as a candidate at that scale
MIN_CONFIDENCE_SCORE_TO_ACCEPT = 0.55  # 0-1 aggregate score, see `_confidence_score`
MAX_AREA_FRACTION_OF_PART = 0.55       # a candidate this large relative to the part is suspect —
                                        # sanity ceiling, not a cap on legitimate large bores (a
                                        # bore up to ~74% of the outer diameter still clears this)
MIN_EDGE_MARGIN_FRACTION = 0.03        # candidate center must be at least this far (as a fraction
                                        # of sqrt(part area), a scale-robust "characteristic size")
                                        # from the part's TRUE outer boundary — not its bounding
                                        # box, which is a poor proxy for a circular/irregular part

# Points/circle centers within this many pixels are treated as the same
# physical feature (both for Hough<->contour cross-checking and final dedup).
DEDUP_DISTANCE_PX = 12.0

# A contour-based hole candidate must be at least this circular
# (4*pi*Area/Perimeter^2, 1.0 = perfect circle) to be trusted at all.
MIN_CIRCULARITY_TO_ACCEPT = MIN_CIRCULARITY  # kept for backwards-compat readability
MIN_CIRCULARITY_FOR_HIGH_CONFIDENCE = 0.90

# Absolute 0-255 gray level considered "genuinely dark" — used by
# `_local_contrast_and_uniformity`'s dark_majority_fraction (see its
# docstring): distinguishes a real cavity with an internal highlight
# (majority dark) from a hollow ring/stroke (minority dark) when both have
# similarly high interior_std.
DARK_PIXEL_THRESHOLD = 60

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
) -> tuple[float, float, float]:
    """Returns (contrast, interior_std, dark_majority_fraction).

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

    But high interior_std ALSO occurs for a real, deep, uniformly-dark hole
    that happens to have a specular highlight in it (confirmed on a real
    polished-metal part: circularity 0.82, solidity 0.98 — clearly a hole
    shape — with interior_std 94, clearly failing the uniformity check on
    its own). `dark_majority_fraction` — the fraction of the interior that's
    genuinely near-black — is what tells these apart: a real cavity's
    interior is MOSTLY dark with a minority bright streak, so this is high;
    a hollow ring/stroke's "filled" interior is mostly the surrounding
    material peeking through its hollow center, so this is low, even though
    both cases can have similarly high interior_std. See MAX_INTERIOR_STD's
    use below for how the two combine.
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
        return 0.0, 255.0, 0.0

    interior_vals = local_gray[interior_mask == 255].astype(np.float64)
    interior_mean = float(np.mean(interior_vals))
    interior_std = float(np.std(interior_vals))
    dark_majority_fraction = float(np.mean(interior_vals < DARK_PIXEL_THRESHOLD))
    ring_mean = cv2.mean(local_gray, mask=ring_mask)[0]
    contrast = abs(interior_mean - ring_mean)
    return contrast, interior_std, dark_majority_fraction


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


CONTOUR_SMOOTHING_EPSILON_FRACTION = 0.01  # same scale as polygon_approx's default,
                                             # already used elsewhere in this module


def _smooth_contour_for_shape_metrics(contour: np.ndarray, epsilon_fraction: float = CONTOUR_SMOOTHING_EPSILON_FRACTION) -> np.ndarray:
    """Removes pixel-level boundary noise — a "scalloped"/jagged contour
    from adaptive-threshold quantization or fine surface texture right at a
    hole's rim (confirmed on a real photo: a genuine, high-solidity [0.96]
    machined bore measured circularity 0.63-0.68 purely from this kind of
    boundary jaggedness) — before circularity/solidity/aspect-ratio are
    computed, so those reflect the true underlying shape rather than a
    sub-pixel artifact. Verified empirically not to matter for a shape
    that's genuinely NOT round: a real elongated/irregular contour's
    circularity stays low after this same smoothing, since the gate is
    about the shape's true form, not its edge noise. Area changes by only
    a few percent — this is not a size measurement change, only a shape
    one. Only used for shape metrics; `_local_contrast_and_uniformity` /
    `_boundary_edge_strength` still sample the ORIGINAL contour, and the
    original (unsmoothed) contour is what's stored on `HoleCandidate` for
    Debug Mode, so nothing about what's shown or sampled at pixel level
    changes — only how "is this circular enough" is judged."""
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return contour
    smoothed = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True)
    return smoothed if len(smoothed) >= 3 else contour


def evaluate_hole_candidates(
    gray: np.ndarray,
    outer_mask: np.ndarray,
    hole_contours: list[np.ndarray],
    hough_hits: list[dict],
    part_area_px2: float,
    outer_contour: np.ndarray | None = None,
) -> list[HoleCandidate]:
    """The actual accept/reject decision. Every raw candidate from
    `segmentation.extract_hole_contours` is evaluated against every gate and
    kept in the returned list REGARDLESS of outcome (with `.accepted` and
    `.rejection_reasons` set) — Vision Debug Mode uses the full list; normal
    measurement uses only the accepted ones via
    `hole_candidates_to_circle_features`.

    `outer_contour` (the part's true outer boundary, not just its bounding
    box) enables an accurate "is this candidate actually well inside the
    part" check — a bounding-box-edge proxy is a poor fit for a circular or
    irregular part, where most of the bbox near its corners isn't part
    material at all. Optional for backward compatibility with callers that
    only have a mask; the position gate is simply skipped when omitted.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    part_scale = math.sqrt(part_area_px2) if part_area_px2 > 0 else 0.0
    min_edge_margin = max(5.0, MIN_EDGE_MARGIN_FRACTION * part_scale)

    hough_used = [False] * len(hough_hits)
    results: list[HoleCandidate] = []

    for c in hole_contours:
        # Shape metrics (circularity/solidity/aspect/equiv-diameter/centroid)
        # are computed from a lightly smoothed copy of the contour — see
        # `_smooth_contour_for_shape_metrics`'s docstring for why. Interior
        # sampling, edge-strength, and everything stored/shown downstream
        # still use the original, unsmoothed `c`.
        c_shape = _smooth_contour_for_shape_metrics(c)
        area, perimeter = area_perimeter(c_shape)
        if perimeter <= 0 or area <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        solidity = _solidity(c_shape, area)
        equiv_diameter = 2.0 * math.sqrt(area / math.pi)

        ellipse = _fit_ellipse_safe(c_shape)
        if ellipse is not None:
            ecx, ecy, major, minor, angle = ellipse
            aspect_ratio = major / minor
        else:
            (ecx, ecy), r = cv2.minEnclosingCircle(c_shape)
            major = minor = 2.0 * r
            angle = 0.0
            aspect_ratio = 1.0

        M = cv2.moments(c_shape)
        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            cx, cy = ecx, ecy

        contrast, interior_std, dark_majority_fraction = _local_contrast_and_uniformity(gray, c, outer_mask)
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
        if interior_std > MAX_INTERIOR_STD and dark_majority_fraction < MIN_DARK_MAJORITY_FRACTION:
            # High variance alone doesn't distinguish a real cavity with an
            # internal highlight from a hollow ring/stroke — both can have
            # it. What tells them apart is whether dark pixels are still the
            # MAJORITY of the interior (a cavity, even a highlighted one) or
            # the minority (a hollow shape's "filled" interior is mostly
            # background peeking through). Only reject when both signals
            # agree something is wrong.
            reasons.append(f"Interior is not a consistent surface (std {interior_std:.1f} > {MAX_INTERIOR_STD:.1f}, "
                            f"and only {dark_majority_fraction:.0%} of it is dark) — looks like a stroke/ring "
                            f"(e.g. text) rather than a filled hole")
        if edge_strength < MIN_EDGE_STRENGTH:
            reasons.append("Weak/soft boundary edges")
        if equiv_diameter < MIN_DIAMETER_PX:
            reasons.append(f"Too small to measure reliably (<{MIN_DIAMETER_PX:.0f}px)")
        if part_area_px2 > 0 and (area / part_area_px2) > MAX_AREA_FRACTION_OF_PART:
            reasons.append(f"Too large relative to the part ({area / part_area_px2:.0%} of part area)")
        if outer_contour is not None:
            edge_margin = distance_to_contour(outer_contour, (cx, cy))
            if edge_margin < min_edge_margin:
                reasons.append(f"Too close to the part's outer edge ({edge_margin:.1f}px margin)")

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


IOU_DEDUP_THRESHOLD = 0.3  # two candidate disks overlapping this much are the same physical hole


def _circle_iou(cx1: float, cy1: float, r1: float, cx2: float, cy2: float, r2: float) -> float:
    """Intersection-over-union of two disks. Needed (rather than plain
    center-distance) because the boundary/rim pathway can produce several
    Hough fits of different radius for the SAME physical hole — e.g. one
    circle tightly on the true rim and another looser one whose center is
    tens of pixels off; a fixed center-distance cutoff misses those, but
    their disks still overlap heavily."""
    d = euclidean_distance((cx1, cy1), (cx2, cy2))
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        inter = math.pi * min(r1, r2) ** 2
    else:
        r1_sq, r2_sq, d_sq = r1 ** 2, r2 ** 2, d ** 2
        alpha = math.acos(float(np.clip((d_sq + r1_sq - r2_sq) / (2 * d * r1), -1.0, 1.0)))
        beta = math.acos(float(np.clip((d_sq + r2_sq - r1_sq) / (2 * d * r2), -1.0, 1.0)))
        inter = r1_sq * (alpha - math.sin(2 * alpha) / 2) + r2_sq * (beta - math.sin(2 * beta) / 2)
    union = math.pi * r1 ** 2 + math.pi * r2 ** 2 - inter
    return inter / union if union > 0 else 0.0


def _dedup_candidates(candidates: list[HoleCandidate]) -> list[HoleCandidate]:
    """Defensive final dedup among ACCEPTED candidates by disk overlap (IoU)
    — the two Otsu partitions that generate region candidates are
    complementary so this rarely triggers for them, but the boundary/rim
    pathway commonly produces several differently-sized Hough fits for the
    same physical hole, which this collapses to the single
    highest-confidence one."""
    accepted_idx = [i for i, c in enumerate(candidates) if c.accepted]
    accepted_idx.sort(key=lambda i: candidates[i].confidence_score, reverse=True)

    kept: list[int] = []
    for i in accepted_idx:
        c = candidates[i]
        c_r = c.equiv_diameter_px / 2.0
        is_dup = False
        for j in kept:
            k = candidates[j]
            k_r = k.equiv_diameter_px / 2.0
            if euclidean_distance((c.cx, c.cy), (k.cx, k.cy)) <= DEDUP_DISTANCE_PX:
                is_dup = True
                break
            if _circle_iou(c.cx, c.cy, c_r, k.cx, k.cy, k_r) >= IOU_DEDUP_THRESHOLD:
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
    outer_contour: np.ndarray | None = None,
) -> tuple[list[CircleFeature], list[HoleCandidate]]:
    """Convenience wrapper tying together Hough cross-checking, full
    candidate evaluation, and conversion to the final reportable list.
    Returns (accepted_circles, all_candidates) — the second is what Vision
    Debug Mode uses to show rejected candidates and why. Pass the part's
    `outer_contour` (e.g. `part.contour`) for an accurate edge-distance
    check on non-rectangular parts."""
    hough_hits = detect_circles_hough(gray, outer_mask, bbox)
    candidates = evaluate_hole_candidates(gray, outer_mask, hole_contours, hough_hits, part_area_px2, outer_contour)
    circles = hole_candidates_to_circle_features(candidates)
    return circles, candidates


def hole_candidates_to_circle_features(candidates: list[HoleCandidate]) -> list[CircleFeature]:
    """The final, reportable result: only candidates that passed every gate,
    converted to the `CircleFeature` shape the rest of the app consumes."""
    features = []
    for i, c in enumerate((x for x in candidates if x.accepted), start=1):
        method = "hough+contour" if c.hough_confirmed else "contour"
        features.append(CircleFeature(
            circle_id=i, cx=c.cx, cy=c.cy, r=c.equiv_diameter_px / 2.0,
            method=method,
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


# --- fork/opening geometry: convex-hull defect analysis ----------------------------
#
# For a fork/C-shaped part (two arms sweeping into a wide opening, like the
# rocker-arm-style bracket this was built against), the entire concave
# inner-arc + hub region is a single deep "defect" relative to the convex
# hull's closing chord between the two outer tip corners. This is what makes
# convexity-defect analysis a reliable, ROTATION-INVARIANT way to find the
# tip corners without assuming any particular part orientation in the photo.

MIN_DEFECT_DEPTH_FRACTION = 0.08   # the largest concavity depth must clear this
                                    # fraction of sqrt(part area) to count as a real
                                    # "opening" rather than ordinary contour noise
MIN_DEFECT_DOMINANCE_RATIO = 1.8   # the largest defect must be at least this many
                                    # times deeper than the second-largest, or which
                                    # concavity is "the" fork opening is ambiguous
MIN_ARC_INLIER_FRACTION = 0.55     # fraction of the concave run that must agree with
                                    # a single circle before trusting an inner-arc fit
MIN_ARC_ANGULAR_SPAN_DEG = 50.0    # a fit covering less than this isn't a meaningful
                                    # "arc diameter", even if the points agree well
ARC_INLIER_TOLERANCE_FRACTION = 0.06


def find_fork_tips(contour: np.ndarray, part_area_px2: float) -> dict | None:
    """Finds the two corners bounding a fork/C-shape's main opening — the
    points where each arm's tip-face meets its inner edge (confirmed
    empirically: the hull's dominant defect starts/ends where the boundary
    turns concave, which is the INNER tip corner, not the outer one — the
    short tip-face run itself stays on the hull all the way out to the
    outer corner). These inner corners are exactly the two points that
    define the fork's clear opening/gap. Returns None — never a guess —
    unless one convexity defect is both deep enough (relative to the
    part's own scale) and clearly dominant over every other concavity; a
    shape that isn't confidently fork-like from this contour should never
    have its "tips" picked from an ambiguous second-largest defect."""
    c = contour.reshape(-1, 1, 2).astype(np.int32)
    if len(c) < 4:
        return None
    hull_idx = cv2.convexHull(c, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return None
    try:
        defects = cv2.convexityDefects(c, hull_idx)
    except cv2.error:
        return None
    if defects is None or len(defects) == 0:
        return None
    defects = defects.reshape(-1, 4)  # OpenCV returns (N,1,4) or (N,4) depending on build

    part_scale = math.sqrt(part_area_px2) if part_area_px2 > 0 else 0.0
    min_depth_px = MIN_DEFECT_DEPTH_FRACTION * part_scale

    depths = defects[:, 3] / 256.0
    order = np.argsort(depths)[::-1]
    largest_depth = float(depths[order[0]])
    second_depth = float(depths[order[1]]) if len(order) > 1 else 0.0

    if largest_depth < min_depth_px:
        return None
    if second_depth > 0 and largest_depth < MIN_DEFECT_DOMINANCE_RATIO * second_depth:
        return None

    start_idx, end_idx, far_idx, _ = [int(v) for v in defects[order[0]]]
    tip_a = c[start_idx][0]
    tip_b = c[end_idx][0]
    far_pt = c[far_idx][0]
    return {
        "tip_a": (float(tip_a[0]), float(tip_a[1])),
        "tip_b": (float(tip_b[0]), float(tip_b[1])),
        "far_point": (float(far_pt[0]), float(far_pt[1])),
        "depth_px": largest_depth,
        "start_idx": start_idx, "end_idx": end_idx, "far_idx": far_idx,
    }


def estimate_part_orientation(
    part: DetectedPart, circles: list[CircleFeature], tips: dict | None
) -> dict:
    """Estimates which way the part is rotated in the photo, for reporting
    and debug visualization only — no measurement in this pipeline depends
    on this (diameters/radii come from fitted circles, distances from
    Euclidean geometry, and Overall Fork Width/Height from `part.rotated_rect`,
    all already rotation-invariant on their own; see `measurement.py`).

    `part.rotated_rect`'s angle alone is ambiguous: a rectangle looks
    identical rotated 180 degrees, and the fit doesn't say which long edge
    is "up". That's resolved here using the vector from the fork-tips'
    midpoint to the main hole's center — both real, already-detected points,
    not a new detection — which gives a full 0-360 degree direction.

    Returns a dict, never a guess:
      - angle_deg: direction (0-360, image coordinates, clockwise from +x)
        from the fork-tip opening toward the main hole, or None if it can't
        be determined confidently.
      - rect_angle_deg: the raw `cv2.minAreaRect` angle (-90, 90), always
        present (it needs no other feature to compute).
      - confidence: "HIGH" when both fork tips and a main hole were found,
        "LOW" otherwise.
      - reason: set when confidence is "LOW", naming what's missing.
      - reference_points: the real points used, for debug visualization.
    """
    _cx, _cy, _rw, _rh, rect_angle = part.rotated_rect

    main_hole = next((c for c in circles if c.feature_type == "bore"), None) or (
        max(circles, key=lambda c: c.r) if circles else None
    )

    if tips is None or main_hole is None:
        missing = []
        if tips is None:
            missing.append("fork tip corners")
        if main_hole is None:
            missing.append("a main hole/bore")
        return {
            "angle_deg": None,
            "rect_angle_deg": rect_angle,
            "confidence": "LOW",
            "reason": (
                "Orientation ambiguous — inspection not evaluated "
                f"(missing: {', '.join(missing)})."
            ),
            "reference_points": {},
        }

    tip_a, tip_b = tips["tip_a"], tips["tip_b"]
    mid = ((tip_a[0] + tip_b[0]) / 2.0, (tip_a[1] + tip_b[1]) / 2.0)
    angle_deg = math.degrees(math.atan2(main_hole.cy - mid[1], main_hole.cx - mid[0])) % 360.0

    return {
        "angle_deg": angle_deg,
        "rect_angle_deg": rect_angle,
        "confidence": "HIGH",
        "reason": "",
        "reference_points": {
            "fork_tip_midpoint": mid,
            "main_hole_center": (main_hole.cx, main_hole.cy),
        },
    }


def _contour_run(contour: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """Contour points from `start_idx` to `end_idx` walking forward, wrapping
    past the array end if needed (the concave run between two convexity-
    defect endpoints, in contour order)."""
    n = len(contour)
    idxs = list(range(start_idx, end_idx + 1)) if end_idx >= start_idx else (
        list(range(start_idx, n)) + list(range(0, end_idx + 1))
    )
    return contour[idxs].reshape(-1, 2).astype(np.float64)


def _circle_from_3_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[float, float, float] | None:
    ax, ay = p1
    bx, by = p2
    cx0, cy0 = p3
    d = 2 * (ax * (by - cy0) + bx * (cy0 - ay) + cx0 * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax ** 2 + ay ** 2) * (by - cy0) + (bx ** 2 + by ** 2) * (cy0 - ay)
          + (cx0 ** 2 + cy0 ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx0 - bx) + (bx ** 2 + by ** 2) * (ax - cx0)
          + (cx0 ** 2 + cy0 ** 2) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    return ux, uy, r


def _angular_span_deg(angles_deg: np.ndarray) -> float:
    """Largest contiguous angular coverage of a set of points on a circle —
    the complement of the largest gap between them, so it's correct even
    when the points wrap past +/-180 degrees (unlike a plain max-min)."""
    a = np.sort(np.mod(angles_deg, 360.0))
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return 360.0 - float(np.max(gaps))


def _ransac_circle_fit(pts: np.ndarray, seed: int = 7) -> dict | None:
    """Robust circle fit shared by `fit_concave_arc` and `fit_convex_arc`.
    Samples 3-point candidate circles (RANSAC-style), scores each by how
    many of the point set's members lie within a tolerance band of it, and
    keeps the best — outliers (a hub bump, a tip-face segment) naturally
    fall out rather than dragging a plain least-squares fit off-target.
    Returns None unless enough points agree with a single circle AND that
    circle covers a real angular span — never a radius guessed from a loose
    or partial fit."""
    n = len(pts)
    if n < 12:
        return None

    # Fixed per-run tolerance (NOT scaled by each candidate's own radius —
    # a degenerate near-collinear 3-point sample yields a huge radius, and
    # scaling tolerance by THAT radius would make the tolerance huge too,
    # spuriously accepting almost every point as an "inlier" of a fake
    # near-straight-line fit). Both derived from the point set's own
    # spatial extent instead.
    run_extent = float(np.max(np.hypot(pts[:, 0] - pts[:, 0].mean(), pts[:, 1] - pts[:, 1].mean()))) * 2 or 1.0
    tol = max(2.0, ARC_INLIER_TOLERANCE_FRACTION * run_extent)
    max_sane_radius = 4.0 * run_extent  # excludes near-collinear degenerate samples outright

    rng = np.random.RandomState(seed)
    best_count = -1
    best_mask = None
    for _ in range(min(80, n * 2)):
        i, j, k = rng.choice(n, size=3, replace=False)
        circle = _circle_from_3_points(pts[i], pts[j], pts[k])
        if circle is None:
            continue
        cx, cy, r = circle
        if r <= 0 or not math.isfinite(r) or r > max_sane_radius:
            continue
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        inliers = np.abs(dists - r) <= tol
        count = int(np.sum(inliers))
        if count > best_count:
            best_count, best_mask = count, inliers

    if best_mask is None or best_count / n < MIN_ARC_INLIER_FRACTION:
        return None

    cx, cy, r, rms = _fit_circle_least_squares(pts[best_mask])
    if r <= 0 or not math.isfinite(r) or rms > 0.12 * r:
        return None

    inlier_pts = pts[best_mask]
    angles = np.degrees(np.arctan2(inlier_pts[:, 1] - cy, inlier_pts[:, 0] - cx))
    angular_span = _angular_span_deg(angles)
    if angular_span < MIN_ARC_ANGULAR_SPAN_DEG:
        return None

    return {
        "cx": float(cx), "cy": float(cy), "radius_px": float(r),
        "inlier_fraction": float(best_count) / n, "angular_span_deg": float(angular_span),
        "rms_px": float(rms), "n_run_points": n,
    }


def fit_concave_arc(contour: np.ndarray, start_idx: int, end_idx: int, seed: int = 7) -> dict | None:
    """Robust circle fit to the concave run between two fork-tip corners
    (the inner-arc side, e.g. Overall Arc / Inner Arc Diameter). That run
    mixes two genuine arc segments with a hub bump in the middle — not one
    consistent circle — which is exactly what `_ransac_circle_fit` is
    built to handle."""
    return _ransac_circle_fit(_contour_run(contour, start_idx, end_idx), seed)


def fit_convex_arc(contour: np.ndarray, start_idx: int, end_idx: int, seed: int = 11) -> dict | None:
    """Robust circle fit to the OUTER (convex) side of the same fork
    contour `fit_concave_arc` fits the inner side of — the complementary
    run from `end_idx` back around to `start_idx` the long way, through the
    true outer arc. Same RANSAC core, same reliability gates; a different
    `seed` from `fit_concave_arc` only to avoid the two fits ever sampling
    an identical candidate sequence by coincidence."""
    return _ransac_circle_fit(_contour_run(contour, end_idx, start_idx), seed)


def measure_tip_thickness(mask: np.ndarray, tip_point: tuple[float, float],
                           reference_center: tuple[float, float], nudge_px: float = 5.0,
                           max_search_px: float = 300.0) -> dict | None:
    """Real material cross-section thickness at a fork tip corner (from
    `find_fork_tips` — the tip corner nearest the part's own concave
    opening). The corner point itself sits exactly on the boundary between
    two arm edges, which makes the "which way is outward" direction too
    sensitive to sub-pixel angular error to start a ray from directly — so
    this first nudges a few pixels INWARD (toward `reference_center`,
    ideally the fitted arc's own center from `fit_concave_arc`, which
    shares its geometry) to land solidly inside the material, then
    ray-marches `mask` OUTWARD from there, counting material pixels until
    it exits into background — a direct measurement of how much material
    is actually there, not an appearance guess. Returns None if the
    direction is degenerate or the nudged start point isn't in material at
    all (nothing invented)."""
    tx, ty = tip_point
    hx, hy = reference_center
    dx, dy = tx - hx, ty - hy
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    dx, dy = dx / norm, dy / norm

    H, W = mask.shape[:2]
    # `tip_point` sits on the INNER-arc side of the boundary (nearest
    # `reference_center`), so material is on the OUTWARD side -- nudge that
    # way first (not toward the center) to reliably land inside it before
    # marching further outward to the true outer edge.
    start_x, start_y = tx + dx * nudge_px, ty + dy * nudge_px
    xi0, yi0 = int(round(start_x)), int(round(start_y))
    if not (0 <= xi0 < W and 0 <= yi0 < H) or mask[yi0, xi0] == 0:
        return None

    steps = 0
    x, y = start_x, start_y
    for _ in range(int(max_search_px)):
        x += dx
        y += dy
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H) or mask[yi, xi] == 0:
            break
        steps += 1

    if steps < 2:
        return None
    return {
        "tip_point_px": (float(tx), float(ty)), "direction": (float(dx), float(dy)),
        "thickness_px": float(steps) + nudge_px,
    }
