"""Batch dataset validation report.

Runs the exact same pipeline `app.py` uses per image (decode -> resize ->
segmentation -> quality gate -> boundary validation -> hole detection ->
measurement -> inspection-spec evaluation -> orientation diagnostic -> QR
scaffolding) across every image in a folder, and produces a per-image plus
aggregate report. Reusable from a script or from the Streamlit app itself.

Never invents a result for an image the pipeline couldn't process: an
image that fails to decode, fails the quality gate, or fails boundary
validation is reported as failed at that exact stage, with the real reason,
not silently skipped or guessed past.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vision import geometry, measurement, preprocessing, qr, segmentation, validation
from vision.inspection_spec import INSPECTION_SPEC
from vision.types import (
    CalibrationProfile,
    PARAM_STATUS_FAIL, PARAM_STATUS_INCOMPLETE, PARAM_STATUS_NOT_DETECTED,
    PARAM_STATUS_NOT_SPECIFIED, PARAM_STATUS_PASS, PARAM_STATUS_SPEC_CONFLICT,
)

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

ALL_PARAM_STATUSES = (
    PARAM_STATUS_PASS, PARAM_STATUS_FAIL, PARAM_STATUS_INCOMPLETE,
    PARAM_STATUS_NOT_DETECTED, PARAM_STATUS_NOT_SPECIFIED, PARAM_STATUS_SPEC_CONFLICT,
)


@dataclass
class ImageReport:
    filename: str
    resolution: tuple[int, int] | None = None  # (width, height)
    loaded: bool = False
    part_detected: bool = False
    boundary_accepted: bool = False
    orientation_angle_deg: float | None = None
    orientation_confidence: str = "LOW"
    orientation_reason: str = ""
    qr_status: str = qr.QR_STATUS_NOT_PRESENT
    processing_time_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # parameter_id -> status ("PASS"/"FAIL"/"INCOMPLETE"/"NOT DETECTED"/
    # "NOT SPECIFIED"/"SPECIFICATION CONFLICT"); empty if the pipeline never
    # reached inspection-spec evaluation for this image.
    parameter_statuses: dict[str, str] = field(default_factory=dict)


def run_single_image(image_path: Path, profile: CalibrationProfile | None = None) -> ImageReport:
    """Runs the full pipeline on one image file. Stops at (and reports) the
    first stage that fails -- exactly like the app's own hard gates."""
    report = ImageReport(filename=image_path.name)
    t0 = time.time()

    try:
        original = preprocessing.decode_image_bytes(image_path.read_bytes())
    except Exception as exc:
        report.errors.append(f"Could not decode image: {exc}")
        report.processing_time_s = time.time() - t0
        return report

    report.loaded = True
    report.resolution = (original.shape[1], original.shape[0])
    img, _ = preprocessing.resize_for_processing(original)

    part = segmentation.build_detected_part(img)
    if part is None:
        report.errors.append("No part boundary detected.")
        report.processing_time_s = time.time() - t0
        return report
    report.part_detected = True

    quality = validation.build_quality_report(img, part, profile)
    for issue in quality.issues:
        (report.errors if issue.severity == "error" else report.warnings).append(issue.message)
    if quality.has_errors():
        report.processing_time_s = time.time() - t0
        return report

    seg_diag = validation.evaluate_segmentation(part, img.shape[:2])
    report.boundary_accepted = seg_diag.accepted
    if not seg_diag.accepted:
        report.errors.extend(seg_diag.reasons)
        report.processing_time_s = time.time() - t0
        return report

    gray = preprocessing.to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    circles = measurement.classify_and_label_circles(circles, part)

    tips = geometry.find_fork_tips(part.contour, part.area_px2)
    orientation = geometry.estimate_part_orientation(part, circles, tips)
    report.orientation_angle_deg = orientation["angle_deg"]
    report.orientation_confidence = orientation["confidence"]
    report.orientation_reason = orientation["reason"]

    report.qr_status = qr.detect_qr(gray)["status"]

    x, y, w, h = part.bbox
    records = measurement.build_measurement_records(part, circles, profile, (float(x), float(y)))
    results = measurement.evaluate_inspection_spec(part, circles, records, profile)
    report.parameter_statuses = {r.parameter_id: r.status for r in results}

    report.processing_time_s = time.time() - t0
    return report


def run_dataset_report(folder: str | Path, profile: CalibrationProfile | None = None) -> dict[str, Any]:
    """Runs `run_single_image` against every supported image file directly
    inside `folder` (non-recursive, sorted by filename) and returns
    {"images": [ImageReport, ...], "aggregate": {...}}."""
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    image_reports = [run_single_image(p, profile) for p in paths]
    return {"images": image_reports, "aggregate": build_aggregate(image_reports)}


def build_aggregate(image_reports: list[ImageReport]) -> dict[str, Any]:
    total = len(image_reports)
    loaded = sum(1 for r in image_reports if r.loaded)
    part_detected = sum(1 for r in image_reports if r.part_detected)
    boundary_accepted = sum(1 for r in image_reports if r.boundary_accepted)
    orientation_high = sum(1 for r in image_reports if r.orientation_confidence == "HIGH")
    orientation_low = sum(1 for r in image_reports if r.loaded and r.orientation_confidence != "HIGH")
    qr_detected = sum(1 for r in image_reports if r.qr_status == qr.QR_STATUS_DETECTED)
    total_time = sum(r.processing_time_s for r in image_reports)

    per_parameter: dict[str, dict[str, int]] = {
        s.parameter_id: {status: 0 for status in ALL_PARAM_STATUSES} for s in INSPECTION_SPEC
    }
    for r in image_reports:
        for pid, status in r.parameter_statuses.items():
            if pid in per_parameter and status in per_parameter[pid]:
                per_parameter[pid][status] += 1

    return {
        "total_images": total,
        "successfully_loaded": loaded,
        "failed_to_load": total - loaded,
        "part_detected": part_detected,
        "boundary_accepted": boundary_accepted,
        "orientation_high_confidence": orientation_high,
        "orientation_low_or_ambiguous": orientation_low,
        "qr_detected": qr_detected,
        "total_processing_time_s": round(total_time, 3),
        "avg_processing_time_s": round(total_time / total, 3) if total else 0.0,
        "per_parameter_status_counts": per_parameter,
    }


def format_report_text(report: dict[str, Any]) -> str:
    """Renders `run_dataset_report`'s result as a plain-text report."""
    lines: list[str] = []
    lines.append("MACHINE VISION DATASET VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    for r in report["images"]:
        lines.append(f"IMAGE: {r.filename}")
        lines.append(f"  Resolution:            {r.resolution}")
        lines.append(f"  Loaded:                {'YES' if r.loaded else 'NO'}")
        lines.append(f"  Part detected:         {'YES' if r.part_detected else 'NO'}")
        lines.append(f"  Boundary accepted:     {'YES' if r.boundary_accepted else 'NO'}")
        if r.orientation_confidence == "HIGH":
            lines.append(f"  Orientation:           {r.orientation_angle_deg:.1f} deg (confidence: HIGH)")
        else:
            lines.append(f"  Orientation:           ambiguous (confidence: LOW) -- {r.orientation_reason}")
        lines.append(f"  QR code:               {r.qr_status}")
        lines.append(f"  Processing time:       {r.processing_time_s:.3f}s")
        if r.parameter_statuses:
            counts: dict[str, int] = {}
            for status in r.parameter_statuses.values():
                counts[status] = counts.get(status, 0) + 1
            breakdown = ", ".join(f"{status}: {n}" for status, n in sorted(counts.items()))
            lines.append(f"  Parameters:            {breakdown}")
        if r.warnings:
            lines.append(f"  Warnings:              {'; '.join(r.warnings)}")
        if r.errors:
            lines.append(f"  Errors:                {'; '.join(r.errors)}")
        lines.append("")

    agg = report["aggregate"]
    lines.append("-" * 60)
    lines.append("AGGREGATE SUMMARY")
    lines.append(f"  TOTAL IMAGES:                {agg['total_images']}")
    lines.append(f"  SUCCESSFULLY LOADED:         {agg['successfully_loaded']}")
    lines.append(f"  FAILED TO LOAD:              {agg['failed_to_load']}")
    lines.append(f"  PART DETECTED:               {agg['part_detected']}")
    lines.append(f"  BOUNDARY ACCEPTED:           {agg['boundary_accepted']}")
    lines.append(f"  ORIENTATION HIGH-CONFIDENCE: {agg['orientation_high_confidence']}")
    lines.append(f"  ORIENTATION LOW/AMBIGUOUS:   {agg['orientation_low_or_ambiguous']}")
    lines.append(f"  QR CODES DETECTED:           {agg['qr_detected']}")
    lines.append(f"  TOTAL PROCESSING TIME:       {agg['total_processing_time_s']:.3f}s")
    lines.append(f"  AVG PROCESSING TIME/IMAGE:   {agg['avg_processing_time_s']:.3f}s")
    lines.append("")
    lines.append("  PER-PARAMETER DETECTION RATES:")
    for pid, counts in agg["per_parameter_status_counts"].items():
        detected = sum(n for status, n in counts.items() if status != PARAM_STATUS_NOT_DETECTED)
        total_seen = sum(counts.values())
        rate = f"{detected}/{total_seen}" if total_seen else "0/0"
        lines.append(f"    {pid:<28} {rate}  ({counts})")

    return "\n".join(lines)
