"""Turns raw quality metrics + detection state into human-readable warnings.

These thresholds are heuristic rules of thumb, not calibrated statistical
limits — they're meant to catch obviously-bad input, not certify good input.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from vision import preprocessing
from vision.types import CalibrationProfile, DetectedPart, QualityIssue, QualityReport, SegmentationDiagnostics

MIN_SHORT_SIDE_PX = 600
BLUR_VARIANCE_THRESHOLD = 100.0
CONTRAST_STD_THRESHOLD = 20.0
MAX_CLIPPED_FRACTION = 0.05

# Hard floors: below these, the image is unusable for ANY measurement, not
# just lower-precision — these are "error" severity (blocking), unlike the
# "warning" thresholds above (which still allow a measurement through).
HARD_MIN_SHORT_SIDE_PX = 150
HARD_BLUR_VARIANCE_THRESHOLD = 15.0

# --- outer-boundary reliability gate -----------------------------------------------
#
# Bounding-rectangle fill ("extent") is NOT used to judge reliability. A
# legitimately concave/open part — a C-bracket, a fork, an L-shape, the
# rocker-arm-style component this was built against — has a lot of
# intentional empty space in its own bounding rectangle by design; a low
# extent means "this part has an opening", not "this segmentation is bad".
# Reliability instead comes from two shape-agnostic signals:
#
#   - solidity (area / convex-hull area): a concavity's hull only "fills in"
#     the concave gap itself, not an arbitrary rectangle, so even a fairly
#     open shape keeps solidity well above pure noise. Measured on a real
#     photo of a valid, but very open (24% bounding-rect fill), forged
#     component: solidity 0.355 — comfortably above the floor below.
#   - compactness (perimeter / (2*sqrt(pi*area)), 1.0 = a circle): a jagged
#     or fragmented boundary has unusually high perimeter for its area
#     regardless of overall shape. Same real component: compactness 2.76.
#
# Both are still just heuristic floors/ceilings, not proofs of correctness —
# see MIN_SOLIDITY_FOR_RELIABLE_CONTOUR / MAX_COMPACTNESS_FOR_RELIABLE_CONTOUR.
MIN_SOLIDITY_FOR_RELIABLE_CONTOUR = 0.15
MAX_COMPACTNESS_FOR_RELIABLE_CONTOUR = 6.0
MIN_PART_AREA_FRACTION_OF_IMAGE = 0.005

# Purely informational threshold for the (non-blocking) extent note in
# `assess_detection` — see that function's docstring.
LOW_EXTENT_INFO_THRESHOLD = 0.55


def assess_image_quality(img: np.ndarray) -> QualityReport:
    gray = preprocessing.to_gray(img)
    h, w = gray.shape

    blur = preprocessing.blur_score(gray)
    contrast = preprocessing.contrast_score(gray)
    dark_frac, bright_frac = preprocessing.exposure_fractions(gray)

    issues: list[QualityIssue] = []

    if min(h, w) < HARD_MIN_SHORT_SIDE_PX:
        issues.append(QualityIssue(
            f"Image resolution ({w}x{h}px) is far too low for any reliable measurement.",
            "error",
        ))
    elif min(h, w) < MIN_SHORT_SIDE_PX:
        issues.append(QualityIssue(
            f"Image resolution is low ({w}x{h}px). Measurement precision will be limited.",
            "warning",
        ))

    if blur < HARD_BLUR_VARIANCE_THRESHOLD:
        issues.append(QualityIssue(
            "Image is far too blurry for any reliable measurement.",
            "error",
        ))
    elif blur < BLUR_VARIANCE_THRESHOLD:
        issues.append(QualityIssue(
            "Image may be too blurry for reliable measurement (low edge sharpness detected).",
            "warning",
        ))

    if contrast < CONTRAST_STD_THRESHOLD:
        issues.append(QualityIssue(
            "Image has low contrast — part edges may be hard to detect accurately.",
            "warning",
        ))

    if dark_frac > MAX_CLIPPED_FRACTION:
        issues.append(QualityIssue(
            "A large portion of the image is very dark (underexposed) — check lighting.",
            "warning",
        ))
    if bright_frac > MAX_CLIPPED_FRACTION:
        issues.append(QualityIssue(
            "A large portion of the image is very bright (overexposed/blown out) — check lighting.",
            "warning",
        ))

    metrics = {
        "width_px": w, "height_px": h,
        "blur_score": blur, "contrast_score": contrast,
        "dark_fraction": dark_frac, "bright_fraction": bright_frac,
    }
    return QualityReport(issues=issues, metrics=metrics)


def evaluate_segmentation(part: DetectedPart | None, image_shape: tuple[int, int]) -> SegmentationDiagnostics:
    """The full outer-boundary accept/reject evidence — see module docstring
    for why bounding-rectangle fill isn't part of the decision. Always
    returns a populated diagnostics object (even for `part is None`) so
    Vision Debug Mode's boundary panel has something to show either way."""
    if part is None:
        return SegmentationDiagnostics(
            area_px2=0.0, perimeter_px=0.0, bbox_area_px2=0.0, rotated_rect_area_px2=0.0,
            extent=0.0, convex_hull_area_px2=0.0, solidity=0.0, compactness=float("inf"),
            area_fraction_of_image=0.0, touches_border=False,
            accepted=False, reasons=["No part boundary could be detected in this image."],
        )

    h, w = image_shape[:2]
    _, _, bw, bh = part.bbox
    bbox_area = float(bw * bh)
    _, _, rw, rh, _ = part.rotated_rect
    rotated_area = rw * rh
    extent = part.area_px2 / rotated_area if rotated_area > 0 else 0.0

    hull = cv2.convexHull(part.contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = part.area_px2 / hull_area if hull_area > 0 else 0.0

    compactness = (
        part.perimeter_px / (2.0 * math.sqrt(math.pi * part.area_px2)) if part.area_px2 > 0 else float("inf")
    )
    area_fraction = part.area_px2 / (h * w) if h * w > 0 else 0.0

    reasons: list[str] = []
    if part.touches_border:
        reasons.append("The part's outline touches the image border, so its true overall "
                        "dimensions are unknown (it may extend beyond the frame). Retake the "
                        "photo with the whole part visible.")
    if area_fraction < MIN_PART_AREA_FRACTION_OF_IMAGE:
        reasons.append("The detected part is too small relative to the image to measure reliably.")
    if solidity < MIN_SOLIDITY_FOR_RELIABLE_CONTOUR:
        reasons.append(
            f"The outline is too fragmented or irregular to trust (solidity {solidity:.0%}, "
            f"expected at least {MIN_SOLIDITY_FOR_RELIABLE_CONTOUR:.0%}). This is not about the "
            f"part's shape — open/concave parts are fine — it means the boundary itself looks "
            f"noisy or broken up."
        )
    if compactness > MAX_COMPACTNESS_FOR_RELIABLE_CONTOUR:
        reasons.append(
            f"The boundary is too jagged or noisy to trust (compactness {compactness:.1f}, "
            f"expected at most {MAX_COMPACTNESS_FOR_RELIABLE_CONTOUR:.1f})."
        )

    return SegmentationDiagnostics(
        area_px2=part.area_px2, perimeter_px=part.perimeter_px,
        bbox_area_px2=bbox_area, rotated_rect_area_px2=float(rotated_area), extent=extent,
        convex_hull_area_px2=hull_area, solidity=solidity, compactness=compactness,
        area_fraction_of_image=area_fraction, touches_border=part.touches_border,
        accepted=not reasons, reasons=reasons,
    )


def segmentation_is_reliable(part: DetectedPart | None, image_shape: tuple[int, int]) -> tuple[bool, str]:
    """Hard gate: refuse to measure at all rather than report numbers built
    on a boundary we don't trust. Thin wrapper around `evaluate_segmentation`
    for callers that just need the yes/no + a message."""
    diag = evaluate_segmentation(part, image_shape)
    return diag.accepted, " ".join(diag.reasons)


def assess_detection(report: QualityReport, part: DetectedPart | None) -> QualityReport:
    if part is None:
        report.issues.append(QualityIssue(
            "Could not confidently detect a part in this image.", "error",
        ))
        return report

    if part.touches_border:
        report.issues.append(QualityIssue(
            "Part may be partially outside the image (its outline touches the image border).",
            "warning",
        ))

    _, _, rw, rh, _ = part.rotated_rect
    rotated_area = rw * rh
    if rotated_area > 0:
        extent = part.area_px2 / rotated_area
        report.metrics["rotated_rect_extent"] = extent
        if extent < LOW_EXTENT_INFO_THRESHOLD:
            # Informational only — NOT a reliability signal (see
            # evaluate_segmentation's docstring). A low fill ratio is normal
            # and expected for open/concave/irregular parts; "info" severity
            # keeps this out of the way unless Advanced mode is on.
            report.issues.append(QualityIssue(
                f"This part's outline fills a relatively small fraction ({extent:.0%}) of its own "
                "bounding rectangle. That's normal for open, concave, or irregular shapes (e.g. a "
                "C/U-bracket or fork) — it is not used to judge detection reliability. If the part "
                "should be roughly rectangular, this could instead indicate perspective distortion "
                "or a segmentation issue worth a visual check.",
                "info",
            ))

    return report


def assess_calibration(report: QualityReport, profile: CalibrationProfile | None,
                        current_image_size: tuple[int, int]) -> QualityReport:
    if profile is None:
        report.issues.append(QualityIssue(
            "Calibration is missing — only pixel measurements are available.", "info",
        ))
        return report

    if profile.method != "perspective" and profile.reference_image_size is not None:
        ref_h, ref_w = profile.reference_image_size
        cur_h, cur_w = current_image_size
        # Resolution mismatch alone isn't fatal (mm/px is a ratio), but a very
        # different aspect ratio/frame suggests the part may not be on the
        # same plane, distance, or camera setup as the calibration reference.
        ref_ratio = ref_w / ref_h if ref_h else 0
        cur_ratio = cur_w / cur_h if cur_h else 0
        if ref_ratio and cur_ratio and abs(ref_ratio - cur_ratio) / ref_ratio > 0.15:
            report.issues.append(QualityIssue(
                "This image's aspect ratio differs noticeably from the calibration reference image. "
                "Confirm the part and reference were captured with the same camera setup and plane.",
                "warning",
            ))

    if profile.method == "known_dimension":
        report.issues.append(QualityIssue(
            "Calibration uses a single known-length reference: this assumes the part and the "
            "reference object are on the same plane and the camera is roughly perpendicular to it. "
            "Perspective distortion is not corrected.",
            "info",
        ))

    return report


def build_quality_report(img: np.ndarray, part: DetectedPart | None,
                          profile: CalibrationProfile | None) -> QualityReport:
    report = assess_image_quality(img)
    report = assess_detection(report, part)
    report = assess_calibration(report, profile, img.shape[:2])
    return report
