"""Shifter-fork recipe evaluation: maps a recipe's attributes to landmarks
computed by `vision.shifter_fork_geometry`, converts to mm via the active
calibration profile (never inventing a scale -- same rule as the rest of
this app), and produces the boolean accept/reject signal.

Only rows with `mapping == "confirmed"` and `in_verdict` True decide that
boolean; "assumed" rows and `in_verdict=False` rows are always computed and
shown (their own PASS/FAIL/INCOMPLETE status, for transparency) but never
contribute to it. Any hard failure in landmark extraction
(`landmarks.reject_reason`) forces `reject=True` regardless of any
individual row -- the "no part / border-touching / no bore / arc-fit-fails"
fail-safe requirement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from vision import calibration, geometry
from vision.shifter_fork_geometry import ShifterForkLandmarks, _point_to_line_distance_px
from vision.shifter_fork_recipes import ShifterForkAttribute, ShifterForkRecipe
from vision.types import (
    CalibrationProfile,
    PARAM_STATUS_FAIL, PARAM_STATUS_INCOMPLETE, PARAM_STATUS_NOT_DETECTED, PARAM_STATUS_PASS,
)


@dataclass
class ShifterForkAttributeResult:
    feature: str
    name: str
    drawing: str
    mapping: str
    in_verdict: bool
    unit: str
    nominal_mm: float
    lower_mm: float
    upper_mm: float
    measured_mm: float | None
    status: str
    reason: str = ""


# --- per-feature px/degree extraction from landmarks ------------------------------

def _px_inner_opening(lm: ShifterForkLandmarks):
    # Sum of each edge's own perpendicular distance to the (independently,
    # more robustly determined) axis, rather than a direct line-to-line
    # distance -- a straight-line fit's own direction has some noise, and
    # measuring straight off the axis for both sides avoids compounding
    # that noise into a single cross-measurement.
    d_left = _px_axis_to_left_inner(lm)
    if lm.right_inner_line is None or lm.axis_origin is None or lm.axis_direction is None:
        return None
    axis_line = {"point": lm.axis_origin, "direction": lm.axis_direction}
    d_right = _point_to_line_distance_px(lm.right_inner_line["point"], axis_line)
    if d_left is None or d_right is None:
        return None
    return d_left + d_right


def _px_axis_to_left_inner(lm: ShifterForkLandmarks):
    if lm.left_inner_line is None or lm.axis_origin is None or lm.axis_direction is None:
        return None
    axis_line = {"point": lm.axis_origin, "direction": lm.axis_direction}
    return _point_to_line_distance_px(lm.left_inner_line["point"], axis_line)


def _px_inner_arc_radius(lm: ShifterForkLandmarks):
    return lm.inner_arc["radius_px"] if lm.inner_arc else None


def _px_axis_to_bore_x(lm: ShifterForkLandmarks):
    if lm.bore is None or lm.axis_origin is None or lm.axis_direction is None:
        return None
    axis_line = {"point": lm.axis_origin, "direction": lm.axis_direction}
    return _point_to_line_distance_px((lm.bore.cx, lm.bore.cy), axis_line)


def _outer_apex(lm: ShifterForkLandmarks):
    arc = lm.outer_arc_left or lm.outer_arc_right
    if arc is None or lm.axis_direction is None:
        return None
    return (arc["cx"] + arc["radius_px"] * lm.axis_direction[0],
            arc["cy"] + arc["radius_px"] * lm.axis_direction[1])


def _px_tip_to_outer_apex(lm: ShifterForkLandmarks):
    apex = _outer_apex(lm)
    if apex is None or lm.tip_point is None:
        return None
    return geometry.euclidean_distance(lm.tip_point, apex)


def _px_tip_to_top(lm: ShifterForkLandmarks):
    if lm.tip_point is None or lm.top_point is None:
        return None
    return geometry.euclidean_distance(lm.tip_point, lm.top_point)


def _px_outer_arc_radius_left(lm: ShifterForkLandmarks):
    return lm.outer_arc_left["radius_px"] if lm.outer_arc_left else None


def _px_outer_arc_radius_right(lm: ShifterForkLandmarks):
    return lm.outer_arc_right["radius_px"] if lm.outer_arc_right else None


def _px_bore_diameter(lm: ShifterForkLandmarks):
    return 2.0 * lm.bore.r if lm.bore else None


def _px_outer_width(lm: ShifterForkLandmarks):
    return lm.outer_width_px


def _px_left_inner_to_bore_x(lm: ShifterForkLandmarks):
    if lm.left_inner_line is None or lm.bore is None:
        return None
    return _point_to_line_distance_px((lm.bore.cx, lm.bore.cy), lm.left_inner_line)


def _px_tip_to_arc_centre(lm: ShifterForkLandmarks):
    arc = lm.outer_arc_left or lm.outer_arc_right
    if arc is None or lm.tip_point is None:
        return None
    return geometry.euclidean_distance(lm.tip_point, (arc["cx"], arc["cy"]))


def _px_left_outer_to_bore_x(lm: ShifterForkLandmarks):
    if lm.bore is None or lm.axis_origin is None or lm.axis_direction is None or lm.outer_left_transverse_px is None:
        return None
    perp = (-lm.axis_direction[1], lm.axis_direction[0])
    bore_transverse = (lm.bore.cx - lm.axis_origin[0]) * perp[0] + (lm.bore.cy - lm.axis_origin[1]) * perp[1]
    return abs(bore_transverse - lm.outer_left_transverse_px)


def _deg_bore_angle(lm: ShifterForkLandmarks):
    if lm.bore is None or lm.axis_origin is None or lm.axis_direction is None:
        return None
    vx, vy = lm.bore.cx - lm.axis_origin[0], lm.bore.cy - lm.axis_origin[1]
    norm = math.hypot(vx, vy)
    if norm < 1e-9:
        return None
    dot = max(-1.0, min(1.0, (vx * lm.axis_direction[0] + vy * lm.axis_direction[1]) / norm))
    return math.degrees(math.acos(dot))


# feature -> px/mm-space extractor (goes through calibration)
_FEATURE_DISPATCH = {
    "inner_opening": _px_inner_opening,
    "axis_to_left_inner": _px_axis_to_left_inner,
    "inner_arc_radius": _px_inner_arc_radius,
    "axis_to_bore_x": _px_axis_to_bore_x,
    "tip_to_outer_apex": _px_tip_to_outer_apex,
    "tip_to_top": _px_tip_to_top,
    "outer_arc_radius_left": _px_outer_arc_radius_left,
    "outer_arc_radius_right": _px_outer_arc_radius_right,
    "bore_diameter": _px_bore_diameter,
    "outer_width": _px_outer_width,
    "left_inner_to_bore_x": _px_left_inner_to_bore_x,
    "tip_to_arc_centre": _px_tip_to_arc_centre,
    "left_outer_to_bore_x": _px_left_outer_to_bore_x,
}

# feature -> degree extractor (no mm calibration involved)
_ANGLE_DISPATCH = {
    "bore_angle_deg": _deg_bore_angle,
}

KNOWN_FEATURES = set(_FEATURE_DISPATCH) | set(_ANGLE_DISPATCH)


def _evaluate_attribute(attr: ShifterForkAttribute, landmarks: ShifterForkLandmarks,
                         profile: CalibrationProfile | None) -> ShifterForkAttributeResult:
    measured = None
    status = PARAM_STATUS_NOT_DETECTED
    reason = ""

    if landmarks.reject_reason is not None:
        reason = landmarks.reject_reason
    elif attr.feature in _ANGLE_DISPATCH:
        value = _ANGLE_DISPATCH[attr.feature](landmarks)
        if value is None:
            reason = "Could not be computed from detected geometry."
        else:
            measured = value
            status = PARAM_STATUS_PASS if attr.min_mm <= measured <= attr.max_mm else PARAM_STATUS_FAIL
    elif attr.feature in _FEATURE_DISPATCH:
        px = _FEATURE_DISPATCH[attr.feature](landmarks)
        if px is None:
            reason = "Feature not detected in this image."
        else:
            mm = calibration.apply_calibration(px, profile)
            if mm is None:
                status = PARAM_STATUS_INCOMPLETE
                reason = "Feature detected in pixels, but no calibration is active to convert to mm."
            else:
                measured = mm
                status = PARAM_STATUS_PASS if attr.min_mm <= measured <= attr.max_mm else PARAM_STATUS_FAIL
    else:
        reason = f"Unknown feature key: {attr.feature!r} (not wired to any landmark)."

    return ShifterForkAttributeResult(
        feature=attr.feature, name=attr.name, drawing=attr.drawing, mapping=attr.mapping,
        in_verdict=attr.in_verdict, unit=attr.unit, nominal_mm=attr.nominal_mm,
        lower_mm=attr.min_mm, upper_mm=attr.max_mm, measured_mm=measured, status=status, reason=reason,
    )


def evaluate_shifter_fork(
    landmarks: ShifterForkLandmarks, recipe: ShifterForkRecipe, profile: CalibrationProfile | None,
) -> tuple[list[ShifterForkAttributeResult], bool, str]:
    results = [_evaluate_attribute(attr, landmarks, profile) for attr in recipe.attributes]

    verdict_rows = [r for r in results if r.mapping == "confirmed" and r.in_verdict]
    failing = [r for r in verdict_rows if r.status != PARAM_STATUS_PASS]

    if landmarks.reject_reason is not None:
        return results, True, landmarks.reject_reason
    if failing:
        bad = failing[0]
        reason = f"{bad.name} {bad.status}" + (f" -- {bad.reason}" if bad.reason else "")
        return results, True, reason
    if not verdict_rows:
        return results, True, "No verdict-deciding (confirmed) attribute could be evaluated."

    return results, False, "All confirmed attributes within tolerance."
