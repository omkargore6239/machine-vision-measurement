"""Turns raw quality metrics + detection state into human-readable warnings.

These thresholds are heuristic rules of thumb, not calibrated statistical
limits — they're meant to catch obviously-bad input, not certify good input.
"""
from __future__ import annotations

import numpy as np

from vision import preprocessing
from vision.types import CalibrationProfile, DetectedPart, QualityIssue, QualityReport

MIN_SHORT_SIDE_PX = 600
BLUR_VARIANCE_THRESHOLD = 100.0
CONTRAST_STD_THRESHOLD = 20.0
MAX_CLIPPED_FRACTION = 0.05
# If the part's rotated-rect extent (area / rotated-rect-area) is below this,
# the shape isn't filling its own minimum rectangle well — could be an
# irregular part, but is also consistent with perspective skew.
LOW_EXTENT_PERSPECTIVE_HINT = 0.55


def assess_image_quality(img: np.ndarray) -> QualityReport:
    gray = preprocessing.to_gray(img)
    h, w = gray.shape

    blur = preprocessing.blur_score(gray)
    contrast = preprocessing.contrast_score(gray)
    dark_frac, bright_frac = preprocessing.exposure_fractions(gray)

    issues: list[QualityIssue] = []

    if min(h, w) < MIN_SHORT_SIDE_PX:
        issues.append(QualityIssue(
            f"Image resolution is low ({w}x{h}px). Measurement precision will be limited.",
            "warning",
        ))

    if blur < BLUR_VARIANCE_THRESHOLD:
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
        if extent < LOW_EXTENT_PERSPECTIVE_HINT:
            report.issues.append(QualityIssue(
                "The detected outline doesn't fill a simple rectangle well — this can indicate "
                "perspective distortion, an irregular part shape, or a noisy segmentation.",
                "warning",
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
