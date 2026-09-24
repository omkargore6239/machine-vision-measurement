"""CSV and text-report builders for downloadable export."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any

from vision.types import CalibrationProfile, MeasurementRecord, QualityReport, ToleranceResult


def measurement_rows(image_name: str, records: list[MeasurementRecord]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        rows.append({
            "Image": image_name,
            "Category": r.category,
            "Feature": r.feature,
            "Pixel Value": "" if r.px_value is None else f"{r.px_value:.3f}",
            "Pixel Unit": r.unit_px,
            "MM Value": "" if r.mm_value is None else f"{r.mm_value:.3f}",
            "MM Unit": r.unit_mm,
            "Confidence": r.confidence,
            "Status": r.status,
        })
    return rows


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def build_report_text(
    image_name: str,
    records: list[MeasurementRecord],
    profile: CalibrationProfile | None,
    quality: QualityReport | None,
    tolerance_results: list[ToleranceResult] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("MACHINE VISION MEASUREMENT REPORT")
    lines.append("=" * 40)
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Image: {image_name}")
    lines.append("")

    lines.append("CALIBRATION")
    lines.append("-" * 40)
    if profile is None:
        lines.append("NOT CALIBRATED — all values below are pixel-only; mm values are unavailable.")
    else:
        lines.append(f"Profile: {profile.name} ({profile.method})")
        lines.append(f"Scale: {profile.mm_per_pixel:.6f} mm/pixel")
        lines.append(f"Confidence: {profile.confidence}")
        for note in profile.notes:
            lines.append(f"  - {note}")
    lines.append("")

    if quality is not None:
        lines.append("IMAGE QUALITY / WARNINGS")
        lines.append("-" * 40)
        if not quality.issues:
            lines.append("No quality issues detected.")
        for issue in quality.issues:
            lines.append(f"  [{issue.severity.upper()}] {issue.message}")
        lines.append("")

    lines.append("MEASUREMENTS")
    lines.append("-" * 40)
    lines.append(f"{'Feature':32s} {'Pixel':>12s} {'MM':>12s} {'Conf':>8s}  Status")
    for r in records:
        px_s = f"{r.px_value:.2f}{r.unit_px}" if r.px_value is not None else "-"
        mm_s = f"{r.mm_value:.3f}{r.unit_mm}" if r.mm_value is not None else "-"
        lines.append(f"{r.feature:32s} {px_s:>12s} {mm_s:>12s} {r.confidence:>8s}  {r.status}")
    lines.append("")

    if tolerance_results:
        lines.append("TOLERANCE CHECKS")
        lines.append("-" * 40)
        for t in tolerance_results:
            measured_s = f"{t.measured_mm:.3f}mm" if t.measured_mm is not None else "n/a"
            lines.append(
                f"{t.feature:32s} nominal={t.nominal_mm:g}mm tol=±{t.tolerance_mm:g}mm "
                f"range=[{t.min_mm:g}, {t.max_mm:g}] measured={measured_s} -> {t.status}"
            )
        lines.append("")

    lines.append("LIMITATIONS")
    lines.append("-" * 40)
    lines.append(
        "This measurement was produced from a single 2D photograph and a heuristic segmentation "
        "pipeline. It does not represent CMM/industrial-metrology-grade accuracy. Actual accuracy "
        "depends on camera resolution, lens distortion, lighting, focus, perspective, calibration "
        "quality, edge quality and part positioning."
    )

    return "\n".join(lines)
