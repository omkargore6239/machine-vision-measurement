"""Assembles pixel geometry + an optional calibration profile into the final
MeasurementRecord table, plus the mm coordinate system and tolerance checks.

This is the single place where px -> mm conversion happens, so it's also the
single place that guarantees the "never fabricate mm" rule: every record's
`mm_value` is None unless a real CalibrationProfile was passed in.
"""
from __future__ import annotations

import itertools

from vision import calibration, geometry
from vision.types import (
    CalibrationProfile, CircleFeature, DetectedPart, MeasurementRecord,
    ToleranceResult, ToleranceSpec,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, MM_UNAVAILABLE_STATUS,
)


def _record(feature: str, category: str, px_value: float | None,
            profile: CalibrationProfile | None, power: int = 1,
            confidence: str = CONFIDENCE_HIGH) -> MeasurementRecord:
    mm_value = None if px_value is None else calibration.apply_calibration(px_value, profile, power)
    unit_mm = "mm" if power == 1 else "mm²"
    status = "Measured" if mm_value is not None else (MM_UNAVAILABLE_STATUS if px_value is not None else "Not detected")
    return MeasurementRecord(
        feature=feature, category=category,
        px_value=px_value, mm_value=mm_value,
        unit_px=("px" if power == 1 else "px²"), unit_mm=unit_mm,
        confidence=confidence if px_value is not None else "N/A",
        status=status,
    )


def resolution_uncertainty_mm(profile: CalibrationProfile | None) -> float | None:
    """A naive lower-bound uncertainty: +/- one pixel's worth of mm at this
    scale. Actual accuracy is also affected by focus, lighting, lens
    distortion and part positioning — this number alone should never be
    presented as the full measurement accuracy."""
    if profile is None or profile.mm_per_pixel is None:
        return None
    return profile.mm_per_pixel


def convert_point_to_mm(px_point: tuple[float, float], origin_px: tuple[float, float],
                         profile: CalibrationProfile | None,
                         flip_y: bool = True) -> tuple[float, float] | None:
    if profile is None or profile.mm_per_pixel is None:
        return None
    dx = px_point[0] - origin_px[0]
    dy = px_point[1] - origin_px[1]
    if flip_y:
        dy = -dy
    return dx * profile.mm_per_pixel, dy * profile.mm_per_pixel


def build_measurement_records(
    part: DetectedPart,
    circles: list[CircleFeature],
    profile: CalibrationProfile | None,
    origin_px: tuple[float, float] | None = None,
) -> list[MeasurementRecord]:
    records: list[MeasurementRecord] = []

    x, y, w, h = part.bbox
    _, _, rw, rh, angle = part.rotated_rect

    records.append(_record("Bounding Box Width", "Overall Dimensions", float(w), profile))
    records.append(_record("Bounding Box Height", "Overall Dimensions", float(h), profile))
    records.append(_record("Rotated Rect Width", "Overall Dimensions", float(rw), profile))
    records.append(_record("Rotated Rect Height", "Overall Dimensions", float(rh), profile))
    records.append(MeasurementRecord(
        feature="Rotated Rect Angle", category="Overall Dimensions",
        px_value=float(angle), mm_value=float(angle),
        unit_px="deg", unit_mm="deg", confidence=CONFIDENCE_HIGH, status="Measured",
    ))
    records.append(_record("Area", "Overall Dimensions", part.area_px2, profile, power=2))
    records.append(_record("Perimeter", "Overall Dimensions", part.perimeter_px, profile))

    origin = origin_px or (float(x), float(y))

    for c in circles:
        prefix = f"Hole {c.circle_id}"
        records.append(_record(f"{prefix} Equivalent Diameter", "Holes", 2 * c.r, profile, confidence=c.confidence))
        records.append(_record(f"{prefix} Radius", "Holes", c.r, profile, confidence=c.confidence))
        if c.major_px is not None and c.minor_px is not None:
            # An ellipse fit exists — report major/minor separately (a
            # circular hole photographed at an angle can appear elliptical;
            # see vision.geometry's hole-detection docstring).
            records.append(_record(f"{prefix} Major Diameter", "Holes", c.major_px, profile, confidence=c.confidence))
            records.append(_record(f"{prefix} Minor Diameter", "Holes", c.minor_px, profile, confidence=c.confidence))

        cx_mm_pt = convert_point_to_mm((c.cx, c.cy), origin, profile)
        cx_val = c.cx - origin[0]
        cy_val = c.cy - origin[1]
        records.append(MeasurementRecord(
            feature=f"{prefix} Center X", category="Holes",
            px_value=cx_val, mm_value=(cx_mm_pt[0] if cx_mm_pt else None),
            unit_px="px", unit_mm="mm", confidence=c.confidence,
            status="Measured" if cx_mm_pt else MM_UNAVAILABLE_STATUS,
        ))
        records.append(MeasurementRecord(
            feature=f"{prefix} Center Y", category="Holes",
            px_value=cy_val, mm_value=(cx_mm_pt[1] if cx_mm_pt else None),
            unit_px="px", unit_mm="mm", confidence=c.confidence,
            status="Measured" if cx_mm_pt else MM_UNAVAILABLE_STATUS,
        ))

        edge_dist_px = geometry.distance_to_contour(part.contour, (c.cx, c.cy))
        records.append(_record(f"{prefix} to Nearest Edge", "Distances", edge_dist_px, profile, confidence=c.confidence))

    for a, b in itertools.combinations(circles, 2):
        dist_px = geometry.euclidean_distance((a.cx, a.cy), (b.cx, b.cy))
        confidence = CONFIDENCE_HIGH if a.confidence == CONFIDENCE_HIGH and b.confidence == CONFIDENCE_HIGH else CONFIDENCE_MEDIUM
        records.append(_record(
            f"Hole {a.circle_id} to Hole {b.circle_id} (center distance)",
            "Distances", dist_px, profile, confidence=confidence,
        ))

    corner_radii = geometry.estimate_corner_radii(part.contour)
    reliable_corners = [c for c in corner_radii if c["reliable"]]
    for i, c in enumerate(reliable_corners, start=1):
        records.append(_record(f"Corner {i} Radius", "Corners & Angles", c["radius_px"], profile))
    if corner_radii and not reliable_corners:
        records.append(MeasurementRecord(
            feature="Corner Radius", category="Corners & Angles",
            px_value=None, mm_value=None, unit_px="px", unit_mm="mm",
            confidence="N/A", status="Radius could not be reliably determined",
        ))

    angles = geometry.estimate_angles(part.contour)
    reliable_angles = [a for a in angles if a["reliable"]]
    for i, a in enumerate(reliable_angles, start=1):
        records.append(MeasurementRecord(
            feature=f"Vertex {i} Angle", category="Corners & Angles",
            px_value=a["angle_deg"], mm_value=a["angle_deg"],
            unit_px="deg", unit_mm="deg", confidence=CONFIDENCE_HIGH, status="Measured",
        ))

    return records


def apply_tolerances(records: list[MeasurementRecord],
                      specs: dict[str, ToleranceSpec]) -> list[ToleranceResult]:
    results = []
    by_feature = {r.feature: r for r in records}
    for feature, spec in specs.items():
        record = by_feature.get(feature)
        min_mm = spec.nominal_mm - spec.tolerance_mm
        max_mm = spec.nominal_mm + spec.tolerance_mm
        measured = record.mm_value if record else None
        if measured is None:
            status = "N/A"
        else:
            status = "PASS" if min_mm <= measured <= max_mm else "FAIL"
        results.append(ToleranceResult(
            feature=feature, nominal_mm=spec.nominal_mm, tolerance_mm=spec.tolerance_mm,
            min_mm=min_mm, max_mm=max_mm, measured_mm=measured, status=status,
        ))
    return results
