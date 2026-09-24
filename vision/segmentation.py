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


def _find_contours_excluding_hole_traces(mask: np.ndarray) -> list[np.ndarray]:
    """Like `cv2.findContours(..., RETR_EXTERNAL, ...)`, except it doesn't
    silently drop a contour that's topologically NESTED inside another
    contour's hierarchy (e.g. a genuine small hole candidate that happens to
    sit inside a thin boundary-bleed "ring" artifact) even though it's
    pixel-wise a completely separate blob — RETR_EXTERNAL only returns
    hierarchy depth 0, hiding anything nested. Confirmed on a real photo:
    a real hole silently vanished this way because it geometrically sat
    inside a 1-2px mask-boundary bleed ring.

    Uses RETR_CCOMP, which re-parents a contour nested inside a HOLE back to
    the top level (parent=-1) — giving us "genuinely external" and
    "nested-inside-a-hole" candidates, while excluding the hole-boundary
    traces themselves (parent != -1), which are usually just a noisier
    near-duplicate of the blob they bound and were found to add spurious
    candidates when tried (plain RETR_LIST) instead of filtered here.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    return [c for c, h in zip(contours, hierarchy[0]) if h[3] == -1]

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

    Requires the FULL requested margin on every side (falls back to the
    unrefined contour otherwise) — confirmed on a real photo where the part
    sat close to the image's top edge: the crop's top margin got clamped to
    the image border, so the local Otsu split had no true "background"
    sample on that side, and a large genuine feature near that edge (a
    machined bore) got misread as outside the part and excluded from the
    mask entirely, silently blinding every downstream hole-candidate method
    to it. A clamped crop can't reliably tell material from background on
    the clamped side, so it isn't safe to trust there regardless of area
    ratio — this must be checked before, not instead of, the area-ratio
    sanity check below.
    """
    x, y, w, h = cv2.boundingRect(contour)
    orig_area = cv2.contourArea(contour)
    if orig_area <= 0:
        return contour

    H, W = gray.shape
    margin = max(10, int(margin_frac * max(w, h)))
    x0, y0 = x - margin, y - margin
    x1, y1 = x + w + margin, y + h + margin
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return contour
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


def _hole_contours_otsu_split(gray: np.ndarray, outer_mask: np.ndarray, min_area_px2: float, max_area_px2: float) -> list[np.ndarray]:
    """A single GLOBAL threshold split of interior pixels into two intensity
    groups; any small enclosed component in either group is a candidate. The
    two groups are complementary (one threshold value), so this never
    double-detects the same region. Weakness: a real photographed surface
    with uneven lighting (e.g. a shading gradient across a metal/plastic
    part) can pull genuine hole-boundary pixels to the "wrong side" of a
    single global threshold, splitting or losing real holes — see the other
    two candidate sources below, which don't share this failure mode."""
    ys, xs = np.where(outer_mask == 255)
    if len(xs) == 0:
        return []
    interior_vals = gray[ys, xs].reshape(-1, 1).astype(np.uint8)
    thresh_val, _ = cv2.threshold(interior_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    high_mask = np.where((gray > thresh_val) & (outer_mask == 255), 255, 0).astype(np.uint8)
    low_mask = np.where((gray <= thresh_val) & (outer_mask == 255), 255, 0).astype(np.uint8)

    contours: list[np.ndarray] = []
    for candidate in (high_mask, low_mask):
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        for c in _find_contours_excluding_hole_traces(candidate):
            area = cv2.contourArea(c)
            if min_area_px2 <= area <= max_area_px2:
                contours.append(c)
    return contours


def _hole_contours_adaptive(gray: np.ndarray, outer_mask: np.ndarray, min_area_px2: float, max_area_px2: float) -> list[np.ndarray]:
    """Locally-adaptive threshold split, restricted to the part's interior.
    Where the global Otsu split above is fooled by a lighting gradient
    across the part's surface, a local threshold that adapts block-by-block
    can still isolate a hole whose LOCAL contrast against its immediate
    surroundings is clear, even if its absolute brightness drifts across the
    part. Block size scales with the part's own size, not a fixed constant."""
    outer_w, outer_h = cv2.boundingRect(outer_mask)[2:]
    block = int(np.clip((min(outer_w, outer_h) // 20) | 1, 15, 101))

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 5,
    )
    masked = cv2.bitwise_and(adaptive, outer_mask)
    masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    masked = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    found = _find_contours_excluding_hole_traces(masked)
    return [c for c in found if min_area_px2 <= cv2.contourArea(c) <= max_area_px2]


def _hole_contours_edge_enclosed(gray: np.ndarray, outer_mask: np.ndarray, min_area_px2: float, max_area_px2: float) -> list[np.ndarray]:
    """Finds regions ENCLOSED by a closed Canny edge loop, independent of
    which side of the edge is brighter. This is the most robust of the three
    methods against lighting gradients (a hole's rim is usually a genuinely
    sharp edge even when neither side has a stable absolute brightness), but
    needs the edge loop to be unbroken, which noisy/low-contrast images can
    defeat — hence using it alongside, not instead of, the threshold-based
    methods above."""
    H, W = gray.shape
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, outer_mask)

    passable = cv2.bitwise_not(edges)
    flood_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    reachable = passable.copy()
    # Seed from image border pixels that are actually passable (non-edge);
    # (0,0) itself may happen to sit on an edge pixel.
    seed = None
    for x, y in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
        if passable[y, x] == 255:
            seed = (x, y)
            break
    if seed is None:
        return []
    cv2.floodFill(reachable, flood_mask, seed, 128)

    enclosed = np.where(reachable == 255, 255, 0).astype(np.uint8)
    enclosed = cv2.bitwise_and(enclosed, outer_mask)

    found = _find_contours_excluding_hole_traces(enclosed)
    return [c for c in found if min_area_px2 <= cv2.contourArea(c) <= max_area_px2]


def _dedup_raw_contours(contours: list[np.ndarray], iou_threshold: float = 0.45) -> list[np.ndarray]:
    """The three methods above frequently propose near-identical contours
    for the same real feature; collapse those down to one (the largest-area
    representative) before they reach evaluation, both for speed and so
    Vision Debug Mode isn't flooded with N copies of the same candidate."""
    boxes = [cv2.boundingRect(c) for c in contours]
    order = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)
    kept: list[int] = []
    for i in order:
        dup = False
        for j in kept:
            if _bbox_iou(boxes[i], boxes[j]) >= iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(i)
    return [contours[i] for i in kept]


def debug_hole_masks(gray: np.ndarray, outer_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Vision Debug Mode only — recomputes the intermediate masks each
    hole-candidate method produces so they can be shown for visual
    inspection. Not used by the actual detection pipeline, which only needs
    the final contours from `extract_hole_contours`."""
    result: dict[str, np.ndarray] = {}
    ys, xs = np.where(outer_mask == 255)
    if len(xs) == 0:
        return result

    interior_vals = gray[ys, xs].reshape(-1, 1).astype(np.uint8)
    thresh_val, _ = cv2.threshold(interior_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result["otsu_dark"] = np.where((gray <= thresh_val) & (outer_mask == 255), 255, 0).astype(np.uint8)
    result["otsu_light"] = np.where((gray > thresh_val) & (outer_mask == 255), 255, 0).astype(np.uint8)

    outer_w, outer_h = cv2.boundingRect(outer_mask)[2:]
    block = int(np.clip((min(outer_w, outer_h) // 20) | 1, 15, 101))
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 5)
    adaptive = cv2.bitwise_and(adaptive, outer_mask)
    result["adaptive_raw"] = adaptive
    morph = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    result["adaptive_after_morphology"] = morph

    edges = cv2.Canny(gray, 40, 120)
    result["canny"] = cv2.bitwise_and(edges, outer_mask)

    return result


def extract_hole_contours(gray: np.ndarray, outer_mask: np.ndarray, min_area_px2: float | None = None) -> list[np.ndarray]:
    """Finds enclosed hole CANDIDATES inside the part — this is a coarse,
    deliberately permissive first pass ("where might a hole be"), not a
    decision about whether something IS a hole. That decision (circularity,
    solidity, contrast, edge strength, position, Hough cross-check,
    confidence scoring) is `geometry.evaluate_hole_candidates`'s job.

    Three independent candidate-generation methods are combined (global Otsu
    split, local adaptive threshold, edge-enclosed regions) for the same
    reason `make_foreground_mask` tries several methods for the outer
    boundary: no single thresholding approach is reliable across the range
    of real lighting conditions, and a single global Otsu split in
    particular is fragile against a shading gradient across the part's
    surface (a common way a real photo's central bore or holes can go
    undetected even though the outer boundary segments fine, since the
    outer-boundary scoring above is a different, more robust process).

    Only a permissive MINIMUM area floor and a generous sanity ceiling are
    applied here. Whether a candidate is too large relative to the part, too
    close to the part's edge, etc. is decided (with a recorded, debuggable
    reason) in `geometry.evaluate_hole_candidates` instead — keeping that
    filtering in one place, visible to Vision Debug Mode.
    """
    if cv2.countNonZero(outer_mask) == 0:
        return []

    outer_area = cv2.countNonZero(outer_mask)
    if min_area_px2 is None:
        min_area_px2 = max(MIN_HOLE_AREA_ABS_PX2, MIN_HOLE_AREA_FRACTION_OF_PART * outer_area)
    max_area_px2 = 0.9 * outer_area  # generous sanity ceiling only; the real limit is in geometry.py

    contours = (
        _hole_contours_otsu_split(gray, outer_mask, min_area_px2, max_area_px2)
        + _hole_contours_adaptive(gray, outer_mask, min_area_px2, max_area_px2)
        + _hole_contours_edge_enclosed(gray, outer_mask, min_area_px2, max_area_px2)
    )
    return _dedup_raw_contours(contours)


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
