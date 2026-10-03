"""Assembles pixel geometry + an optional calibration profile into the final
MeasurementRecord table, plus the mm coordinate system and tolerance checks.

This is the single place where px -> mm conversion happens, so it's also the
single place that guarantees the "never fabricate mm" rule: every record's
`mm_value` is None unless a real CalibrationProfile was passed in.
"""
from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

from vision import calibration, geometry
from vision.inspection_spec import INSPECTION_SPEC
from vision.types import (
    CalibrationProfile, CircleFeature, DetectedPart, MeasurementRecord, ParameterResult,
    ToleranceDefinition, ToleranceResult, ToleranceSpec, bilateral,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, MM_UNAVAILABLE_STATUS,
    PARAM_STATUS_FAIL, PARAM_STATUS_INCOMPLETE, PARAM_STATUS_NOT_DETECTED, PARAM_STATUS_NOT_SPECIFIED,
    PARAM_STATUS_PASS, PARAM_STATUS_SPEC_CONFLICT,
)

# A circular feature is classified as the "Center Bore" (rather than just
# another "Hole N") when it's both clearly the largest and sits near the
# part's centroid — heuristic thresholds, not a CAD-aware feature
# classifier. With only one circular feature present, "clearly the largest"
# is trivially true, so a single large centered hole is still called the
# bore (spec: a lone central bore must still be recognized as one).
BORE_MIN_DIAMETER_FRACTION_OF_PART = 0.15   # vs sqrt(part area), a scale-robust size proxy
BORE_MIN_CENTROID_PROXIMITY_FRACTION = 0.18  # vs sqrt(part area)
BORE_MIN_SIZE_RATIO_VS_OTHERS = 1.4


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


def classify_and_label_circles(circles: list[CircleFeature], part: DetectedPart) -> list[CircleFeature]:
    """Labels each detected circular feature as "Hole N" or "Center Bore"
    (see module-level constants for the heuristic). Geometry detection has
    no notion of "which one is the bore" — that's a reporting/labeling
    concern, which is why this lives here rather than in vision.geometry.
    Mutates and returns the same list."""
    if not circles:
        return circles

    M = cv2.moments(part.contour)
    if M["m00"] > 0:
        centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    else:
        x, y, w, h = part.bbox
        centroid = (x + w / 2, y + h / 2)

    part_scale = math.sqrt(part.area_px2) if part.area_px2 > 0 else 0.0

    diameters = [2 * c.r for c in circles]
    max_d = max(diameters)
    max_idx = diameters.index(max_d)
    others = [d for i, d in enumerate(diameters) if i != max_idx]
    median_other = sorted(others)[len(others) // 2] if others else None

    bore_idx = None
    if part_scale > 0:
        candidate = circles[max_idx]
        dist_to_centroid = geometry.euclidean_distance((candidate.cx, candidate.cy), centroid)
        near_center = dist_to_centroid <= BORE_MIN_CENTROID_PROXIMITY_FRACTION * part_scale
        large_enough = max_d >= BORE_MIN_DIAMETER_FRACTION_OF_PART * part_scale
        clearly_largest = median_other is None or (
            median_other > 0 and max_d / median_other >= BORE_MIN_SIZE_RATIO_VS_OTHERS
        )
        if near_center and large_enough and clearly_largest:
            bore_idx = max_idx

    hole_n = 0
    for i, c in enumerate(circles):
        if i == bore_idx:
            c.feature_type, c.label, c.short_id = "bore", "Center Bore", "CB"
        else:
            hole_n += 1
            c.feature_type, c.label, c.short_id = "hole", f"Hole {hole_n}", f"H{hole_n}"

    return circles


def build_feature_summary_table(
    circles: list[CircleFeature],
    profile: CalibrationProfile | None,
    origin_px: tuple[float, float],
    tolerance_results: list[ToleranceResult] | None = None,
) -> list[dict]:
    """The compact per-feature table the spec asks for: one row per circular
    feature with X/Y/diameter/confidence/tolerance/result together, as
    opposed to `build_measurement_records`'s long format (one row per
    attribute, used for the full detailed table and CSV export)."""
    tol_by_feature = {t.feature: t for t in (tolerance_results or [])}
    rows = []
    for c in circles:
        label = c.label or f"Hole {c.circle_id}"
        diameter_px = 2 * c.r
        diameter_mm = calibration.apply_calibration(diameter_px, profile)
        pt_mm = convert_point_to_mm((c.cx, c.cy), origin_px, profile)

        if pt_mm is not None:
            x_val, y_val, unit, diam_val = pt_mm[0], pt_mm[1], "mm", diameter_mm
        else:
            x_val, y_val, unit = c.cx - origin_px[0], c.cy - origin_px[1], "px"
            diam_val = diameter_px

        tol = tol_by_feature.get(f"{label} Equivalent Diameter")
        rows.append({
            "Feature": c.short_id or label,
            "Type": "Central Bore" if c.feature_type == "bore" else "Hole",
            "X": round(x_val, 3),
            "Y": round(y_val, 3),
            "Diameter": round(diam_val, 3) if diam_val is not None else None,
            "Unit": unit,
            "Confidence": c.confidence,
            "Tolerance": f"±{tol.tolerance_mm:g}" if tol else "—",
            "Result": tol.status if tol else "N/A",
        })
    return rows


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
        prefix = c.label or f"Hole {c.circle_id}"
        category = "Center Bore" if c.feature_type == "bore" else "Holes"
        records.append(_record(f"{prefix} Equivalent Diameter", category, 2 * c.r, profile, confidence=c.confidence))
        records.append(_record(f"{prefix} Radius", category, c.r, profile, confidence=c.confidence))
        if c.major_px is not None and c.minor_px is not None:
            # An ellipse fit exists — report major/minor separately (a
            # circular hole photographed at an angle can appear elliptical;
            # see vision.geometry's hole-detection docstring).
            records.append(_record(f"{prefix} Major Diameter", category, c.major_px, profile, confidence=c.confidence))
            records.append(_record(f"{prefix} Minor Diameter", category, c.minor_px, profile, confidence=c.confidence))

        cx_mm_pt = convert_point_to_mm((c.cx, c.cy), origin, profile)
        cx_val = c.cx - origin[0]
        cy_val = c.cy - origin[1]
        records.append(MeasurementRecord(
            feature=f"{prefix} Center X", category=category,
            px_value=cx_val, mm_value=(cx_mm_pt[0] if cx_mm_pt else None),
            unit_px="px", unit_mm="mm", confidence=c.confidence,
            status="Measured" if cx_mm_pt else MM_UNAVAILABLE_STATUS,
        ))
        records.append(MeasurementRecord(
            feature=f"{prefix} Center Y", category=category,
            px_value=cy_val, mm_value=(cx_mm_pt[1] if cx_mm_pt else None),
            unit_px="px", unit_mm="mm", confidence=c.confidence,
            status="Measured" if cx_mm_pt else MM_UNAVAILABLE_STATUS,
        ))

        edge_dist_px = geometry.distance_to_contour(part.contour, (c.cx, c.cy))
        records.append(_record(f"{prefix} to Nearest Edge", "Distances", edge_dist_px, profile, confidence=c.confidence))

        # Straight vertical (image-Y) distance from this feature's centre down
        # to the part's lowest contour point.
        lowest_y = float(part.contour[:, 0, 1].max())
        records.append(_record(
            f"{prefix} Center to Lowest Point (vertical)", "Distances",
            max(0.0, lowest_y - c.cy), profile, confidence=c.confidence,
        ))

    # Perpendicular distance from each feature centre to the line through the
    # two fork tips (the part's own base line) -- the hole's height in the
    # part's frame, independent of how the photo is rotated.
    tips = geometry.find_fork_tips(part.contour, part.area_px2)
    if tips is not None:
        ta, tb = np.array(tips["tip_a"], float), np.array(tips["tip_b"], float)
        base = tb - ta
        base_len = float(np.hypot(*base))
        if base_len > 1e-6:
            for c in circles:
                prefix = c.label or f"Hole {c.circle_id}"
                perp = abs(base[0] * (c.cy - ta[1]) - base[1] * (c.cx - ta[0])) / base_len
                records.append(_record(
                    f"{prefix} Center to Tip Line (perpendicular)", "Distances", perp, profile,
                    confidence=c.confidence,
                ))

    for a, b in itertools.combinations(circles, 2):
        dist_px = geometry.euclidean_distance((a.cx, a.cy), (b.cx, b.cy))
        confidence = CONFIDENCE_HIGH if a.confidence == CONFIDENCE_HIGH and b.confidence == CONFIDENCE_HIGH else CONFIDENCE_MEDIUM
        label_a = a.label or f"Hole {a.circle_id}"
        label_b = b.label or f"Hole {b.circle_id}"
        records.append(_record(
            f"{label_a} to {label_b} (center distance)",
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


# --- product-specific inspection spec (fork bracket) --------------------------------
#
# Pure orchestration: calls the already-gated geometry functions
# (`geometry.find_fork_tips` / `fit_concave_arc` / `measure_tip_thickness`,
# plus the existing hole/dimension pipeline) and applies the
# PASS/FAIL/INCOMPLETE/NOT-DETECTED rules. No new detection logic lives
# here — see `vision.geometry` for the actual fork-geometry algorithms and
# `vision.inspection_spec` for the parameter definitions/nominal values.

def _median_tip_thickness(mask, run, ref_center, from_start: bool) -> dict | None:
    """Radial arm thickness near one fork tip, as the median over several
    points a little way along the arm. A single point too close to the tip
    corner can lie on the flat end face of the arm, where the radial ray
    runs along the face and reads a few pixels instead of the real
    thickness; further along (beyond the end face) the reading is stable.
    Inset points scale with the contour-run length so this works at any
    image resolution. Returns the measurement dict whose thickness is the
    median, or None when no point gave a reliable reading."""
    n = len(run)
    if n < 16:
        return None
    found = []
    for frac in (0.07, 0.09, 0.11, 0.13):
        i = max(1, int(n * frac))
        pt = tuple(run[i] if from_start else run[-i - 1])
        t = geometry.measure_tip_thickness(mask, pt, ref_center)
        if t is not None:
            found.append(t)
    if not found:
        return None
    found.sort(key=lambda t: t["thickness_px"])
    return found[len(found) // 2]


def _fork_tip_geometry(part: DetectedPart) -> dict:
    """Computes the shared fork-geometry evidence once (tips, inner-arc
    fit, outer-arc fit, per-tip thickness) so every parameter that needs it
    reuses the same result instead of recomputing. Every entry is None when
    its own reliability gate wasn't cleared — never a guess."""
    tips = geometry.find_fork_tips(part.contour, part.area_px2)
    arc_fit = convex_arc_fit = None
    thickness_a = thickness_b = None
    if tips is not None:
        arc_fit = geometry.fit_concave_arc(part.contour, tips["start_idx"], tips["end_idx"])
        convex_arc_fit = geometry.fit_convex_arc(part.contour, tips["start_idx"], tips["end_idx"])
        if arc_fit is not None:
            ref_center = (arc_fit["cx"], arc_fit["cy"])
            run = geometry._contour_run(part.contour, tips["start_idx"], tips["end_idx"])
            thickness_a = _median_tip_thickness(part.mask, run, ref_center, from_start=True)
            thickness_b = _median_tip_thickness(part.mask, run, ref_center, from_start=False)
    return {
        "tips": tips, "arc_fit": arc_fit, "convex_arc_fit": convex_arc_fit,
        "thickness_a": thickness_a, "thickness_b": thickness_b,
    }


def evaluate_inspection_spec(
    part: DetectedPart,
    circles: list[CircleFeature],
    records: list[MeasurementRecord],
    profile: CalibrationProfile | None,
    tolerance_overrides: dict[str, tuple[float, float]] | None = None,
) -> list[ParameterResult]:
    """Evaluates every parameter in `vision.inspection_spec.INSPECTION_SPEC`
    against this part's actual detected geometry.

    `tolerance_overrides` maps a `parameter_id` to an operator-configured
    `(nominal_mm, tolerance_mm)` pair (always bilateral — the spec's own
    positive-only/negative-only/none tolerances are used until the operator
    explicitly overrides one). An override on a conflict-flagged parameter
    counts as the "engineering confirmation" the conflict is waiting for —
    it resolves that one parameter for this image; without an override, a
    conflict is NEVER auto-resolved.

    Status rules (seven states, not four):
      - NOT DETECTED: the underlying feature/geometry wasn't found at all
        (or the spec itself documents no measurable definition — see each
        `ParameterSpec.not_detectable_reason`).
      - INCOMPLETE: a real pixel measurement exists but no calibration is
        active to convert it to mm, or the parameter is flagged as never
        measurable from a single 2D photo (Part Thickness).
      - NOT SPECIFIED: a real mm measurement exists but no approved
        tolerance does — never guessed into a PASS or FAIL.
      - SPECIFICATION CONFLICT: two candidate specs disagree on the nominal
        (e.g. Overall Fork Width) and neither has been explicitly confirmed
        via an override — the measured value is still shown, but no
        PASS/FAIL is ever computed from it.
      - PASS / FAIL: a real mm measurement exists and a real tolerance
        (spec-defined or operator-overridden) says which side of the line
        it's on.
    """
    overrides = tolerance_overrides or {}
    fork = _fork_tip_geometry(part)
    tips, arc_fit, convex_arc_fit = fork["tips"], fork["arc_fit"], fork["convex_arc_fit"]
    thickness_a, thickness_b = fork["thickness_a"], fork["thickness_b"]

    sorted_circles = sorted(circles, key=lambda c: c.r, reverse=True)
    main_hole = next((c for c in circles if c.feature_type == "bore"), None) or (
        sorted_circles[0] if sorted_circles else None
    )
    boss_hole = next((c for c in sorted_circles if c is not main_hole), None)

    rec_by_feature = {r.feature: r for r in records}
    results: list[ParameterResult] = []

    for spec in INSPECTION_SPEC:
        override = overrides.get(spec.parameter_id)
        if override is not None:
            nominal_mm, tolerance_def = override[0], bilateral(override[1])
        else:
            nominal_mm, tolerance_def = spec.nominal_mm, spec.tolerance
        lower_mm, upper_mm = tolerance_def.limits(nominal_mm)

        measured_px: float | None = None
        confidence = "N/A"
        method = ""
        reason = ""
        conflict_note = ""
        debug: dict = {}

        # --- SPECIFICATION CONFLICT: never auto-resolved, unless the
        #     operator has explicitly overridden this parameter (that IS
        #     the engineering confirmation the conflict was waiting for). --
        if spec.conflicting_spec is not None and override is None:
            other = spec.conflicting_spec
            conflict_note = (
                f"Existing parameter: {spec.nominal_mm:g} {spec.tolerance.display()}  |  "
                f"Additional engineering list: {other.nominal_mm:g} {other.tolerance.display()}  |  "
                f"Engineering confirmation required."
            )
            reason = "Two candidate specifications disagree on the nominal value. Override in Tolerance Configuration to confirm one."

        if spec.requires_second_view:
            method = "N/A -- requires a second camera/view or 3D measurement"
            reason = ("Current 2D image does not provide reliable thickness geometry along the "
                      "camera's depth axis. Requires a side-view camera, a second camera, a "
                      "calibrated side image, or 3D measurement.")
            results.append(ParameterResult(
                spec.parameter_id, spec.name, spec.characteristic_type, spec.value_kind, nominal_mm,
                tolerance_def.display(), lower_mm, upper_mm, None, "mm", None,
                PARAM_STATUS_INCOMPLETE, confidence, method, reason, conflict_note, debug,
            ))
            continue

        if spec.not_detectable_reason is not None:
            results.append(ParameterResult(
                spec.parameter_id, spec.name, spec.characteristic_type, spec.value_kind, nominal_mm,
                tolerance_def.display(), lower_mm, upper_mm, None, "mm", None,
                PARAM_STATUS_NOT_DETECTED, confidence, "not yet mapped to a documented geometric definition",
                spec.not_detectable_reason, conflict_note, debug,
            ))
            continue

        if spec.parameter_id == "main_hole_diameter":
            method = "existing hole-detection pipeline (geometry.detect_holes) -- the classified bore, or largest accepted circle"
            if main_hole is not None:
                measured_px = 2 * main_hole.r
                confidence = main_hole.confidence
                debug = {"center_px": (main_hole.cx, main_hole.cy), "diameter_px": measured_px, "feature_type": main_hole.feature_type}
            else:
                reason = "No circular hole feature passed the existing detection gates."

        elif spec.parameter_id == "top_boss_hole_diameter":
            method = "existing hole-detection pipeline -- second-largest accepted circle, distinct from the main hole"
            if boss_hole is not None:
                measured_px = 2 * boss_hole.r
                confidence = boss_hole.confidence
                debug = {"center_px": (boss_hole.cx, boss_hole.cy), "diameter_px": measured_px}
            else:
                reason = "A second circular hole, distinct from the main hole, was not reliably detected."

        elif spec.parameter_id == "overall_fork_width":
            # Rotated Rect Width/Height (part.rotated_rect, cv2.minAreaRect)
            # are the part's OWN minimum-area bounding rectangle -- tied to
            # the part's shape, not the image's x/y axes, and already
            # normalized rw >= rh (geometry.rotated_rect). The axis-aligned
            # Bounding Box Width/Height swap values if the same part is
            # photographed rotated 90 degrees; the rotated-rect ones don't.
            method = "existing Rotated Rect Width record (part's own minimum bounding rectangle -- orientation-independent, unlike the axis-aligned bounding box)"
            r = rec_by_feature.get("Rotated Rect Width")
            if r is not None and r.px_value is not None:
                measured_px = r.px_value
                confidence = CONFIDENCE_HIGH
                debug = {"source_record": "Rotated Rect Width"}
            else:
                reason = "Rotated-rect width was not available."

        elif spec.parameter_id == "overall_height":
            method = "existing Rotated Rect Height record (part's own minimum bounding rectangle -- orientation-independent, unlike the axis-aligned bounding box)"
            r = rec_by_feature.get("Rotated Rect Height")
            if r is not None and r.px_value is not None:
                measured_px = r.px_value
                confidence = CONFIDENCE_HIGH
                debug = {"source_record": "Rotated Rect Height"}
            else:
                reason = "Rotated-rect height was not available."

        elif spec.parameter_id == "hole_center_distance":
            method = "euclidean distance between the two detected hole centers"
            if main_hole is not None and boss_hole is not None:
                measured_px = geometry.euclidean_distance((main_hole.cx, main_hole.cy), (boss_hole.cx, boss_hole.cy))
                confidence = (CONFIDENCE_HIGH if main_hole.confidence == CONFIDENCE_HIGH and boss_hole.confidence == CONFIDENCE_HIGH
                              else CONFIDENCE_MEDIUM)
                debug = {"main_hole_center_px": (main_hole.cx, main_hole.cy), "boss_hole_center_px": (boss_hole.cx, boss_hole.cy)}
            else:
                reason = "Both the main hole and top boss hole must be reliably detected to compute center-to-center distance."

        elif spec.parameter_id == "inner_arc_diameter":
            method = "RANSAC circle fit to the concave inner-arc contour run (geometry.fit_concave_arc)"
            if tips is None:
                reason = "The part's contour doesn't show a single, clearly dominant concave opening (fork shape not confidently detected)."
            elif arc_fit is None:
                reason = "The inner-arc region didn't fit a single circle tightly or widely enough to trust (hub geometry or noise dominates)."
            else:
                measured_px = 2 * arc_fit["radius_px"]
                confidence = CONFIDENCE_HIGH if arc_fit["inlier_fraction"] >= 0.75 else CONFIDENCE_MEDIUM
                debug = {
                    "fit_center_px": (arc_fit["cx"], arc_fit["cy"]), "radius_px": arc_fit["radius_px"],
                    "inlier_fraction": arc_fit["inlier_fraction"], "angular_span_deg": arc_fit["angular_span_deg"],
                    "rms_px": arc_fit["rms_px"],
                }

        elif spec.parameter_id == "fork_tip_gap":
            method = "distance between the two fork tip corners (geometry.find_fork_tips)"
            if tips is None:
                reason = "Fork tip corners were not reliably identified from the contour (no single, dominant concave opening found)."
            else:
                measured_px = geometry.euclidean_distance(tips["tip_a"], tips["tip_b"])
                confidence = CONFIDENCE_HIGH
                debug = {"tip_a_px": tips["tip_a"], "tip_b_px": tips["tip_b"], "defect_depth_px": tips["depth_px"]}

        elif spec.parameter_id == "fork_arm_length":
            method = "distance from each fork tip corner to the main hole center, averaged (documented anchor: tip -> main hole center)"
            if tips is None or main_hole is None:
                reason = "Requires both reliable fork tips and a detected main hole."
            else:
                d_a = geometry.euclidean_distance(tips["tip_a"], (main_hole.cx, main_hole.cy))
                d_b = geometry.euclidean_distance(tips["tip_b"], (main_hole.cx, main_hole.cy))
                measured_px = (d_a + d_b) / 2
                confidence = CONFIDENCE_MEDIUM
                debug = {"arm_a_px": d_a, "arm_b_px": d_b, "anchor": "main hole center"}

        elif spec.parameter_id == "fork_tip_thickness":
            method = "mask ray-march outward from just inside each tip, perpendicular to the inner arc (geometry.measure_tip_thickness)"
            vals = [t["thickness_px"] for t in (thickness_a, thickness_b) if t is not None]
            if not vals:
                reason = "Could not reliably measure material thickness at either fork tip."
            else:
                measured_px = sum(vals) / len(vals)
                confidence = CONFIDENCE_HIGH if len(vals) == 2 else CONFIDENCE_MEDIUM
                debug = {"tip_a": thickness_a, "tip_b": thickness_b}

        elif spec.parameter_id in ("outer_arc_radius", "arc_radius"):
            method = "RANSAC circle fit to the convex outer-arc contour run (geometry.fit_convex_arc)"
            if tips is None:
                reason = "The part's contour doesn't show a single, clearly dominant concave opening (fork shape not confidently detected)."
            elif convex_arc_fit is None:
                reason = "The outer-arc region didn't fit a single circle tightly or widely enough to trust."
            else:
                measured_px = convex_arc_fit["radius_px"]  # value_kind="radius" -- report the radius directly, never doubled
                confidence = CONFIDENCE_HIGH if convex_arc_fit["inlier_fraction"] >= 0.75 else CONFIDENCE_MEDIUM
                debug = {
                    "fit_center_px": (convex_arc_fit["cx"], convex_arc_fit["cy"]), "radius_px": convex_arc_fit["radius_px"],
                    "inlier_fraction": convex_arc_fit["inlier_fraction"], "angular_span_deg": convex_arc_fit["angular_span_deg"],
                    "rms_px": convex_arc_fit["rms_px"],
                    "note": "Same fitted outer-arc geometry as 'Outer Arc Radius'/'Arc Radius' -- the two given nominal "
                            "candidates (62.00 vs 62.75) were not distinguished by any documented geometric difference.",
                }

        measured_mm: float | None = None
        if measured_px is not None:
            measured_mm = calibration.apply_calibration(measured_px, profile)
            debug["pixel_measurement_px"] = measured_px
            if profile is not None and profile.mm_per_pixel is not None:
                debug["calibration_mm_per_px"] = profile.mm_per_pixel

        if conflict_note:
            status = PARAM_STATUS_SPEC_CONFLICT
        elif measured_px is None:
            status = PARAM_STATUS_NOT_DETECTED
        elif measured_mm is None:
            status = PARAM_STATUS_INCOMPLETE
            reason = reason or "Feature detected in pixels, but no calibration is active to convert to mm."
        elif not tolerance_def.has_tolerance:
            status = PARAM_STATUS_NOT_SPECIFIED
            reason = reason or "No approved tolerance exists for this characteristic -- measured value shown for reference only."
        else:
            status = PARAM_STATUS_PASS if lower_mm <= measured_mm <= upper_mm else PARAM_STATUS_FAIL

        deviation_mm = (measured_mm - nominal_mm) if measured_mm is not None else None

        results.append(ParameterResult(
            spec.parameter_id, spec.name, spec.characteristic_type, spec.value_kind, nominal_mm,
            tolerance_def.display(), lower_mm, upper_mm, measured_mm, "mm", deviation_mm, status,
            confidence, method, reason, conflict_note, debug,
        ))

    return results
