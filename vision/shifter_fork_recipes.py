"""Shifter Fork inspection recipes -- ported verbatim from the working Flask
prototype's part-number spec sheet. Each attribute is tagged "confirmed"
(unambiguous from the drawing -- decides the accept/reject verdict) or
"assumed" (best-guess mapping, shown for reference only, pending sign-off),
and can carry `in_verdict=False` to force a row to stay informational
regardless of mapping (used for the bore -- its H7 tolerance is tighter
than a camera can resolve, so it's reported but never judged).

`feature` is the dispatch key into `vision.shifter_fork_geometry`'s
landmarks (see `vision.shifter_fork_measurement`'s dispatch table).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShifterForkAttribute:
    feature: str
    name: str
    drawing: str
    nominal_mm: float
    min_mm: float
    max_mm: float
    mapping: str  # "confirmed" | "assumed"
    unit: str = "mm"  # "mm" | "deg"
    in_verdict: bool = True  # False forces info-only regardless of mapping
    needs_view: str = ""     # non-empty -> not measurable from the top view (e.g. "side view"); never judged


@dataclass
class ShifterForkRecipe:
    part_number: str
    title: str
    attributes: list[ShifterForkAttribute] = field(default_factory=list)


def _side_view_attributes(thickness_nominal: float) -> list[ShifterForkAttribute]:
    """MAJOR characteristics on every fork drawing that are only visible in
    the side/section views (B-B / D-D / Detail B), never in a top-down photo.
    Listed so they are not silently forgotten; they stay "NEEDS SIDE VIEW"
    until a side camera or gauge result is available, and never affect the
    accept/reject verdict."""
    t = thickness_nominal
    return [
        ShifterForkAttribute("cross_hole_5h10", "Cross Hole Diameter", "Ø5 H10 (+0.048/0), thru",
                              5.0, 5.0, 5.048, "confirmed", in_verdict=False, needs_view="side view"),
        ShifterForkAttribute("pad_thickness", "Pad Thickness", f"{t:g} ±0.05",
                              t, t - 0.05, t + 0.05, "confirmed", in_verdict=False, needs_view="side view"),
        ShifterForkAttribute("tip_thickness_side", "Tip Thickness (Detail B)", "7.4 ±0.1",
                              7.4, 7.3, 7.5, "confirmed", in_verdict=False, needs_view="side view"),
    ]


SHIFTER_FORK_RECIPES: dict[str, ShifterForkRecipe] = {
    "OP-T02-013-A-00-010": ShifterForkRecipe(
        part_number="OP-T02-013-A-00-010",
        title="Shifter Fork 1st / 2nd Speed",
        attributes=[
            ShifterForkAttribute("inner_opening", "Inner Fork Opening", "96 +0.3/0",
                                  96.0, 96.0, 96.3, "confirmed"),
            ShifterForkAttribute("axis_to_left_inner", "Fork Half Opening", "48 ±0.15",
                                  48.0, 47.85, 48.15, "confirmed"),
            ShifterForkAttribute("inner_arc_radius", "Inner Arc Radius", "R48",
                                  48.0, 47.0, 49.0, "confirmed"),
            ShifterForkAttribute("axis_to_bore_x", "Bore Offset from Axis", "17.68 ±0.15",
                                  17.68, 17.53, 17.83, "assumed"),
            ShifterForkAttribute("tip_to_outer_apex", "Arc Height", "70.1 ±0.2",
                                  70.1, 69.9, 70.3, "assumed"),
            ShifterForkAttribute("tip_to_top", "Overall Height", "90.1",
                                  90.1, 89.8, 90.4, "assumed"),
            ShifterForkAttribute("outer_arc_radius_left", "Outer Arc Radius (left)", "R80 +0.5/0",
                                  80.0, 80.0, 80.5, "assumed"),
            ShifterForkAttribute("bore_angle_deg", "Bore Angular Position", "28.4° ±20'",
                                  28.4, 28.067, 28.733, "assumed", unit="°"),
            ShifterForkAttribute("bore_diameter", "Bore Diameter", "Ø14 H7",
                                  14.0, 14.0, 14.018, "confirmed", in_verdict=False),
        ] + _side_view_attributes(13.0),
    ),
    "OP-T02-023-A-00-010": ShifterForkRecipe(
        part_number="OP-T02-023-A-00-010",
        title="Shifter Fork 3rd / 4th Speed",
        attributes=[
            ShifterForkAttribute("outer_width", "Overall Fork Width", "104 ±0.3",
                                  104.0, 103.7, 104.3, "confirmed"),
            ShifterForkAttribute("inner_opening", "Inner Fork Opening", "94.2 +0.3/0",
                                  94.2, 94.2, 94.5, "confirmed"),
            ShifterForkAttribute("axis_to_left_inner", "Fork Half Opening", "47.1 ±0.15",
                                  47.1, 46.95, 47.25, "confirmed"),
            ShifterForkAttribute("left_inner_to_bore_x", "Reference Distance", "40.8 ±0.15",
                                  40.8, 40.65, 40.95, "assumed"),
            ShifterForkAttribute("tip_to_top", "Overall/Arc Height", "68.6 ±0.15",
                                  68.6, 68.45, 68.75, "assumed"),
            ShifterForkAttribute("tip_to_arc_centre", "Arc Centre Offset", "0.5 +0.2/0",
                                  0.5, 0.5, 0.7, "assumed"),
            ShifterForkAttribute("outer_arc_radius_left", "Outer Arc Radius (left)", "R71 +0.5/0",
                                  71.0, 71.0, 71.5, "assumed"),
            ShifterForkAttribute("outer_arc_radius_right", "Outer Arc Radius (right)", "R87 +0.5/0",
                                  87.0, 87.0, 87.5, "assumed"),
            ShifterForkAttribute("bore_diameter", "Bore Diameter", "Ø14 H7",
                                  14.0, 14.0, 14.018, "confirmed", in_verdict=False),
        ] + _side_view_attributes(12.7),
    ),
    "OP-T02-015-A-00-010": ShifterForkRecipe(
        part_number="OP-T02-015-A-00-010",
        title="Shifter Fork 5th / 6th Speed",
        attributes=[
            ShifterForkAttribute("outer_width", "Overall Fork Width", "104 ±0.3",
                                  104.0, 103.7, 104.3, "confirmed"),
            ShifterForkAttribute("inner_opening", "Inner Fork Opening", "94.2 +0.3/0",
                                  94.2, 94.2, 94.5, "confirmed"),
            ShifterForkAttribute("axis_to_left_inner", "Fork Half Opening", "47.1 ±0.15",
                                  47.1, 46.95, 47.25, "confirmed"),
            ShifterForkAttribute("inner_arc_radius", "Inner Arc Radius", "R47.1",
                                  47.1, 46.1, 48.1, "confirmed"),
            ShifterForkAttribute("left_inner_to_bore_x", "Reference Distance", "27.65 ±0.15",
                                  27.65, 27.5, 27.8, "assumed"),
            ShifterForkAttribute("left_outer_to_bore_x", "Bore Position from Left", "41.9",
                                  41.9, 41.6, 42.2, "assumed"),
            ShifterForkAttribute("tip_to_top", "Overall/Arc Height", "66.1 ±0.15",
                                  66.1, 65.95, 66.25, "assumed"),
            ShifterForkAttribute("tip_to_outer_apex", "Related Arc Height", "65.6 +0.2/0",
                                  65.6, 65.6, 65.8, "assumed"),
            ShifterForkAttribute("outer_arc_radius_left", "Outer Arc Radius (left)", "R62.75",
                                  62.75, 61.75, 63.75, "assumed"),
            ShifterForkAttribute("outer_arc_radius_right", "Outer Arc Radius (right)", "R62",
                                  62.0, 61.0, 63.0, "assumed"),
            ShifterForkAttribute("bore_diameter", "Bore Diameter", "Ø14 H7",
                                  14.0, 14.0, 14.018, "confirmed", in_verdict=False),
        ] + _side_view_attributes(12.7),
    ),
}
