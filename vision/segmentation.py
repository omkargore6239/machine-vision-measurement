"""Part detection: turn a raw image into a foreground mask + outer contour.

No single threshold works for every image, so several candidate masks are
generated (Otsu, inverse Otsu, adaptive threshold, Canny+morphology, connected
components) and scored; the best-scoring plausible object is kept. This is a
heuristic prototype-grade segmenter, not a certified industrial one — see
`vision.validation` for the quality warnings shown alongside its output.
"""
from __future__ import annotations

import cv2
import numpy as np

from vision.preprocessing import to_gray
from vision.types import DetectedPart
from vision import geometry

# Candidates covering less than this fraction of the image, or more than the
# upper fraction, are rejected as implausible (too small to be "the part", or
# so large it's probably the whole background/frame).
MIN_OBJECT_AREA_FRACTION = 0.01
MAX_OBJECT_AREA_FRACTION = 0.95

# Hole-candidate area floor is a FRACTION of the part's own area, not a fixed
# pixel count. A fixed absolute floor (e.g. 15px^2) is essentially noise-level
# on a high-resolution real photo — every fleck of JPEG noise, fine texture,
# or a text serif clears it, which is how a real product photo can produce
# hundreds of false "holes". Scaling by part size keeps the floor meaningful
# regardless of image resolution or how tightly the part fills the frame.
MIN_HOLE_AREA_FRACTION_OF_PART = 0.0003
MIN_HOLE_AREA_ABS_PX2 = 12.0


def _score_contour(c: np.ndarray, image_area: float, W: int, H: int) -> float:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    rect_area = w * h
    if rect_area <= 0:
        return -1.0
    extent = area / rect_area
    center_x, center_y = x + w / 2, y + h / 2
    center_penalty = abs(center_x - W / 2) / (W / 2) + abs(center_y - H / 2) / (H / 2)
    return (area / image_area) * 4 + extent * 1.5 - center_penalty * 0.25


def _candidate_masks(gray: np.ndarray) -> list[tuple[np.ndarray, str]]:
    candidates = []

    for mode, name in [(cv2.THRESH_BINARY, "otsu"), (cv2.THRESH_BINARY_INV, "otsu_inv")]:
        _, m = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
        candidates.append((m, name))

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7
    )
    candidates.append((adaptive, "adaptive_threshold"))

    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    candidates.append((edges, "canny_morphology"))

    # Connected components on the inverse-Otsu mask: pick the largest
    # non-background component as an alternative to contour-area scoring.
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(otsu_inv, connectivity=8)
    if num_labels > 1:
        # label 0 is background; pick the largest remaining component.
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        cc_mask = np.where(labels == largest, 255, 0).astype(np.uint8)
        candidates.append((cc_mask, "connected_components"))

    return candidates


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _refine_contour_locally(gray: np.ndarray, contour: np.ndarray, margin_frac: float = 0.15) -> np.ndarray:
    """The coarse multi-method scoring above picks a plausible candidate, but
    some methods (e.g. a large fixed-size adaptive-threshold block) can bleed
    several pixels past the true edge on flat regions. This tightens the
    boundary by re-thresholding (Otsu) a small crop around the coarse
    detection, where a single local threshold is almost always accurate.
    Falls back to the original contour if the refinement looks implausible
    (wrong polarity, nothing found, wildly different size).
    """
    x, y, w, h = cv2.boundingRect(contour)
    orig_area = cv2.contourArea(contour)
    if orig_area <= 0:
        return contour

    H, W = gray.shape
    margin = max(10, int(margin_frac * max(w, h)))
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
    crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (5, 5), 0)

    best_candidate = None
    best_iou = -1.0
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, m = cv2.threshold(crop, 0, 255, mode + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        c_full = c + np.array([[x0, y0]])
        iou = _bbox_iou(cv2.boundingRect(c_full), (x, y, w, h))
        if iou > best_iou:
            best_iou = iou
            best_candidate = c_full

    if best_candidate is None or best_iou < 0.5:
        return contour

    refined_area = cv2.contourArea(best_candidate)
    if refined_area < 0.5 * orig_area or refined_area > 1.8 * orig_area:
        return contour

    return best_candidate


def make_foreground_mask(img: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """Returns (mask, method_name) for the best-scoring plausible object, or
    (None, None) if nothing plausible was found."""
    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    H, W = blurred.shape
    image_area = float(H * W)

    best_contour = None
    best_score = -1.0
    best_method = None

    for mask, name in _candidate_masks(blurred):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < image_area * MIN_OBJECT_AREA_FRACTION or area > image_area * MAX_OBJECT_AREA_FRACTION:
                continue
            score = _score_contour(c, image_area, W, H)
            if score > best_score:
                best_score = score
                best_contour = c
                best_method = name

    if best_contour is None:
        return None, None

    best_contour = _refine_contour_locally(gray, best_contour)

    final_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(final_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return final_mask, best_method


def extract_outer_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def extract_hole_contours(
    gray: np.ndarray, outer_mask: np.ndarray, min_area_px2: float | None = None, border_margin: int = 2,
) -> list[np.ndarray]:
    """Finds enclosed hole CANDIDATES inside the part — this is a coarse,
    deliberately permissive first pass ("where might a hole be"), not a
    decision about whether something IS a hole. That decision (circularity,
    solidity, contrast, edge strength, Hough cross-check, confidence scoring)
    is `geometry.evaluate_hole_candidates`'s job — see its docstring.

    `outer_mask` (from `make_foreground_mask`) is solid-filled — it has no
    interior holes of its own, by design, since it's used for outer-shape
    geometry and for "is this point inside the part" checks. Holes are
    instead found by re-examining the grayscale image restricted to the
    part's interior: Otsu splits interior pixels into two intensity groups,
    and any small, fully-enclosed (non-border-touching) connected component
    in either group is a hole candidate. The two groups are complementary
    (one threshold value), so this never double-detects the same region.
    """
    ys, xs = np.where(outer_mask == 255)
    if len(xs) == 0:
        return []

    interior_vals = gray[ys, xs].reshape(-1, 1).astype(np.uint8)
    thresh_val, _ = cv2.threshold(interior_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    outer_x, outer_y, outer_w, outer_h = cv2.boundingRect(outer_mask)
    outer_area = cv2.countNonZero(outer_mask)

    if min_area_px2 is None:
        min_area_px2 = max(MIN_HOLE_AREA_ABS_PX2, MIN_HOLE_AREA_FRACTION_OF_PART * outer_area)

    high_mask = np.where((gray > thresh_val) & (outer_mask == 255), 255, 0).astype(np.uint8)
    low_mask = np.where((gray <= thresh_val) & (outer_mask == 255), 255, 0).astype(np.uint8)

    holes: list[np.ndarray] = []
    for candidate in (high_mask, low_mask):
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_px2 or area > 0.5 * outer_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            touches = (
                bx <= outer_x + border_margin or by <= outer_y + border_margin
                or (bx + bw) >= (outer_x + outer_w - border_margin)
                or (by + bh) >= (outer_y + outer_h - border_margin)
            )
            if touches:
                continue
            holes.append(c)

    return holes


def touches_border(contour: np.ndarray, img_shape: tuple[int, int], margin: int = 2) -> bool:
    H, W = img_shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    return x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin)


def build_detected_part(img: np.ndarray) -> DetectedPart | None:
    """Runs segmentation + outer/hole contour extraction and assembles a
    DetectedPart. Returns None if no plausible part was found."""
    mask, method = make_foreground_mask(img)
    if mask is None:
        return None

    outer = extract_outer_contour(mask)
    if outer is None:
        return None

    area, perimeter = geometry.area_perimeter(outer)
    bbox = geometry.bounding_box(outer)
    rrect = geometry.rotated_rect(outer)
    holes = extract_hole_contours(to_gray(img), mask)

    return DetectedPart(
        contour=outer,
        mask=mask,
        segmentation_method=method or "unknown",
        bbox=bbox,
        rotated_rect=rrect,
        area_px2=area,
        perimeter_px=perimeter,
        hole_contours=holes,
        touches_border=touches_border(outer, img.shape),
    )
