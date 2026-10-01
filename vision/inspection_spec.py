"""Product-specific inspection configuration for the fork/rocker-arm
bracket used to validate this app against a real part photo.

`INSPECTION_SPEC` is the ACTIVE, evaluated set -- exactly the original 10
parameters, per the operator's explicit 2026-09-30 request ("measure only
these 10"). Their nominal/tolerance values are the operator-confirmed
"Tentative Nominal" table from that date (92.00/105.00/72.00/82.00/8.00 mm
etc.) -- a real reference, not a placeholder, though still tentative
pending final engineering approval. `vision.measurement.evaluate_inspection_spec`
iterates this list directly, so anything not in it is never measured or
shown.

`ADDITIONAL_SHEET_SPEC` holds 12 more parameters from a later engineering
sheet, layered on top earlier this session -- kept here (not deleted) but
deliberately EXCLUDED from `INSPECTION_SPEC`/active evaluation per the same
2026-09-30 request, in case they're wanted again. Most of them were already
`not_detectable_reason`-gated even when active (no documented anchor to
compute them from -- see each entry's comment), so excluding them changes
nothing about whether they were ever actually measured, only whether
they're evaluated/displayed at all.

The operator/admin can override nominal/tolerance for any active parameter
from the Inspection Settings panel. This module only holds the defaults and
the mapping telling `evaluate_inspection_spec` how to compute each
parameter -- no measurement logic lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from vision.types import (
    CHARACTERISTIC_ARM_LENGTH, CHARACTERISTIC_FORK_TIP_GAP, CHARACTERISTIC_HOLE_DIAMETER,
    CHARACTERISTIC_HOLE_PITCH, CHARACTERISTIC_INNER_ARC_DIAMETER, CHARACTERISTIC_OVERALL_HEIGHT,
    CHARACTERISTIC_OVERALL_WIDTH, CHARACTERISTIC_RADIUS, CHARACTERISTIC_THICKNESS,
    CHARACTERISTIC_TIP_THICKNESS, ToleranceDefinition, bilateral, not_specified,
)


@dataclass(frozen=True)
class ParameterSpec:
    parameter_id: str
    name: str
    characteristic_type: str
    value_kind: str  # "diameter" | "radius" | "linear"
    nominal_mm: Optional[float]
    tolerance: ToleranceDefinition
    requires_second_view: bool = False  # never measurable from one 2D photo, regardless of image content
    not_detectable_reason: Optional[str] = None  # set -> orchestrator short-circuits to NOT DETECTED
    conflicting_spec: Optional["ParameterSpec"] = None  # set -> orchestrator always reports SPECIFICATION CONFLICT


# --- the additional-sheet's conflicting candidate for Overall Fork Width --
# Not part of either list above -- it's the second candidate for #4,
# referenced directly by that entry below. Kept regardless of
# ADDITIONAL_SHEET_SPEC's active/inactive state since #4 (Overall Fork
# Width) is itself always active.
_FORK_WIDTH_CONFLICT_CANDIDATE = ParameterSpec(
    "overall_fork_width", "Overall Fork Width", CHARACTERISTIC_OVERALL_WIDTH, "linear",
    104.00, bilateral(0.30),
)

# --- active spec: the original 10, per the operator's 2026-09-30 confirmed
#     "Tentative Nominal" table -------------------------------------------
INSPECTION_SPEC: list[ParameterSpec] = [
    ParameterSpec("inner_arc_diameter", "Overall Arc / Inner Arc Diameter",
                   CHARACTERISTIC_INNER_ARC_DIAMETER, "diameter", 92.00, bilateral(0.50)),
    ParameterSpec("main_hole_diameter", "Main Hole Diameter",
                   CHARACTERISTIC_HOLE_DIAMETER, "diameter", 20.00, bilateral(0.20)),
    ParameterSpec("top_boss_hole_diameter", "Top Boss Hole Diameter",
                   CHARACTERISTIC_HOLE_DIAMETER, "diameter", 16.00, bilateral(0.20)),
    # Nominal conflicts with the additional engineering sheet (104.00±0.30) —
    # explicitly flagged, never auto-resolved. See _FORK_WIDTH_CONFLICT_CANDIDATE.
    ParameterSpec("overall_fork_width", "Overall Fork Width",
                   CHARACTERISTIC_OVERALL_WIDTH, "linear", 105.00, bilateral(0.30),
                   conflicting_spec=_FORK_WIDTH_CONFLICT_CANDIDATE),
    ParameterSpec("fork_arm_length", "Fork Arm Length",
                   CHARACTERISTIC_ARM_LENGTH, "linear", 48.00, bilateral(0.50)),
    ParameterSpec("fork_tip_gap", "Fork Tip Gap / Opening",
                   CHARACTERISTIC_FORK_TIP_GAP, "linear", 72.00, bilateral(0.50)),
    ParameterSpec("overall_height", "Overall Height",
                   CHARACTERISTIC_OVERALL_HEIGHT, "linear", 82.00, bilateral(0.50)),
    ParameterSpec("part_thickness", "Part Thickness",
                   CHARACTERISTIC_THICKNESS, "linear", 12.00, bilateral(0.20), requires_second_view=True),
    ParameterSpec("hole_center_distance", "Hole Center-to-Center Distance",
                   CHARACTERISTIC_HOLE_PITCH, "linear", 78.00, bilateral(0.30)),
    ParameterSpec("fork_tip_thickness", "Fork Tip Thickness",
                   CHARACTERISTIC_TIP_THICKNESS, "linear", 8.00, bilateral(0.20)),
]

# --- inactive: the additional engineering sheet (12-21c) -- kept, not
#     evaluated. Reactivate by appending (some or all of) this list back
#     onto INSPECTION_SPEC. ---------------------------------------------
ADDITIONAL_SHEET_SPEC: list[ParameterSpec] = [
    # 12: distinct from both Inner Arc Diameter (92.00) and Fork Tip Gap
    # (72.00) -- no documented anchor says what third measurement this is.
    ParameterSpec("inner_fork_opening", "Inner Fork Opening",
                   CHARACTERISTIC_INNER_ARC_DIAMETER, "linear", 94.20, ToleranceDefinition("positive_only", plus_mm=0.30),
                   not_detectable_reason=("Ambiguous relative to Overall Arc/Inner Arc Diameter (92.00mm) and Fork Tip "
                                           "Gap (72.00mm) -- neither matches, and no documented reference points "
                                           "distinguish a third measurement. Requires engineering drawing to clarify.")),
    # 13: not literally half of Overall Fork Width (105/104 -> 52.5/52.0,
    # neither is 47.10) -- not a derivable quantity, and no anchor given.
    ParameterSpec("fork_half_width", "Fork Half Width",
                   CHARACTERISTIC_OVERALL_WIDTH, "linear", 47.10, bilateral(0.15),
                   not_detectable_reason=("Not derivable as half of Overall Fork Width (would be 52.5 or 52.0mm, not "
                                           "47.10mm) and no independent anchor is documented.")),
    # 14: the name alone gives no geometric hint at all.
    ParameterSpec("reference_distance", "Reference Distance",
                   CHARACTERISTIC_HOLE_PITCH, "linear", 27.65, bilateral(0.15),
                   not_detectable_reason="No documented reference points -- the parameter name alone doesn't identify which two features this spans."),
    # 15/16: distinct from Overall Height (82.00) -- likely "just the arc
    # portion's height", but the cutoff/reference points aren't documented.
    ParameterSpec("overall_arc_height", "Overall / Arc Height",
                   CHARACTERISTIC_OVERALL_HEIGHT, "linear", 66.00, bilateral(0.15),
                   not_detectable_reason=("Distinct from Overall Height (82.00mm) -- likely a sub-region height (e.g. "
                                           "just the arc, excluding the hub/boss), but which reference points define "
                                           "it isn't documented.")),
    ParameterSpec("related_arc_height", "Related Arc Height",
                   CHARACTERISTIC_OVERALL_HEIGHT, "linear", 65.60, ToleranceDefinition("positive_only", plus_mm=0.20),
                   not_detectable_reason="0.40mm from Overall/Arc Height -- reads as a second reference point on the same undocumented height definition."),
    # 17: a genuinely different radius from the existing inner-arc fit
    # (~R46, matching item 1's 92.00mm diameter) -- a second arc with no
    # independent way to locate it on the contour.
    ParameterSpec("inner_arc_radius", "Inner Arc Radius",
                   CHARACTERISTIC_RADIUS, "radius", 57.10, ToleranceDefinition("positive_only", plus_mm=0.50),
                   not_detectable_reason=("Doesn't match the existing inner-arc fit (~R46.0, i.e. the same geometry "
                                           "as the 92.00mm Overall Arc/Inner Arc Diameter) -- would require locating "
                                           "a second, independent arc feature not otherwise identified on this contour.")),
    # 18/19: the one pair with a clean, reusable geometric construction --
    # see geometry.fit_convex_arc. No approved tolerance for either -> both
    # always report NOT SPECIFIED even though a real value is measured.
    ParameterSpec("outer_arc_radius", "Outer Arc Radius",
                   CHARACTERISTIC_RADIUS, "radius", 62.00, not_specified()),
    ParameterSpec("arc_radius", "Arc Radius",
                   CHARACTERISTIC_RADIUS, "radius", 62.75, not_specified()),
    # 20: geometry.estimate_corner_radii typically finds MULTIPLE reliable
    # corners on this part -- "Transition Radius" (singular) doesn't say
    # which one, and guessing (e.g. "the largest") is an invented definition.
    ParameterSpec("transition_radius", "Transition Radius",
                   CHARACTERISTIC_RADIUS, "radius", 14.50, bilateral(0.50),
                   not_detectable_reason=("Multiple reliable corner radii are typically detected on this contour "
                                           "(see the Geometry table) -- which one is \"the\" Transition Radius isn't "
                                           "documented, and picking one (e.g. the largest) would be an invented "
                                           "definition, not a measurement.")),
    # 21a-c: no anchors at all; the brief itself only asks to split these
    # into separate characteristics "if the geometry/image provides enough
    # information" -- it doesn't.
    ParameterSpec("boss_upper_geometry_a", "Boss / Upper Geometry (a)",
                   CHARACTERISTIC_THICKNESS, "linear", 12.00, not_specified(),
                   not_detectable_reason="No documented anchor/geometry definition for any of the three Boss/Upper Geometry values."),
    ParameterSpec("boss_upper_geometry_b", "Boss / Upper Geometry (b)",
                   CHARACTERISTIC_THICKNESS, "linear", 6.00, not_specified(),
                   not_detectable_reason="No documented anchor/geometry definition for any of the three Boss/Upper Geometry values."),
    ParameterSpec("boss_upper_geometry_c", "Boss / Upper Geometry (c)",
                   CHARACTERISTIC_THICKNESS, "linear", 13.00, not_specified(),
                   not_detectable_reason="No documented anchor/geometry definition for any of the three Boss/Upper Geometry values."),
]


def get_spec(parameter_id: str) -> Optional[ParameterSpec]:
    return next((p for p in INSPECTION_SPEC if p.parameter_id == parameter_id), None)
