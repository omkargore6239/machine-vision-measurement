"""Landmark-accuracy tests for `vision.shifter_fork_geometry`, against the
synthetic `make_shifter_fork_part` fixture's exact ground truth, plus the
four explicit fail-safe conditions from the brief: no part detected, part
touching the image border, no bore found, and an arc fit failing -- each
must set `reject_reason` (never a silent guess past a hard failure)."""
import math

import cv2
import numpy as np
import pytest

from tests.fixtures import make_shifter_fork_part
from vision import geometry, segmentation
from vision.preprocessing import to_gray
from vision.shifter_fork_geometry import extract_shifter_fork_landmarks

# Pixel-level tolerance for landmark accuracy against the fixture's exact
# ground truth -- generous enough to absorb real (small) fit noise from the
# windowed RANSAC/robust-line fits, tight enough to catch a real regression.
TOL_PX = 3.0
TOL_FRACTION = 0.03  # 3% for radii/distances measured in tens-to-hundreds of px


def _build():
    img, gt = make_shifter_fork_part()
    part = segmentation.build_detected_part(img)
    assert part is not None
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    lm = extract_shifter_fork_landmarks(part, circles)
    return lm, gt


def test_no_reject_on_a_clean_synthetic_part():
    lm, gt = _build()
    assert lm.reject_reason is None


def test_tip_point_matches_ground_truth():
    lm, gt = _build()
    assert geometry.euclidean_distance(lm.tip_point, gt["tip_point"]) <= TOL_PX


def test_top_point_matches_ground_truth():
    lm, gt = _build()
    assert geometry.euclidean_distance(lm.top_point, gt["top_point"]) <= TOL_PX


def test_inner_arc_radius_matches_ground_truth():
    lm, gt = _build()
    assert lm.inner_arc is not None
    assert abs(lm.inner_arc["radius_px"] - gt["inner_arc_radius_px"]) <= gt["inner_arc_radius_px"] * TOL_FRACTION


def test_outer_arc_radius_matches_ground_truth():
    lm, gt = _build()
    assert lm.outer_arc_left is not None
    assert abs(lm.outer_arc_left["radius_px"] - gt["outer_arc_radius_px"]) <= gt["outer_arc_radius_px"] * TOL_FRACTION


def test_outer_width_matches_ground_truth():
    lm, gt = _build()
    assert lm.outer_width_px is not None
    assert abs(lm.outer_width_px - gt["outer_width_px"]) <= TOL_PX


def test_inner_edges_are_near_vertical_and_symmetric():
    lm, gt = _build()
    assert lm.left_inner_line is not None and lm.right_inner_line is not None
    # this fixture's true inner edges are exactly parallel to the axis
    for line in (lm.left_inner_line, lm.right_inner_line):
        dx, dy = line["direction"]
        angle_from_axis_deg = math.degrees(math.atan2(abs(dx), abs(dy)))
        assert angle_from_axis_deg <= 3.0


def test_bore_matches_ground_truth():
    lm, gt = _build()
    assert lm.bore is not None
    assert geometry.euclidean_distance((lm.bore.cx, lm.bore.cy), gt["bore_center"]) <= TOL_PX
    assert abs(lm.bore.r - gt["bore_radius_px"]) <= gt["bore_radius_px"] * 0.15


def test_axis_direction_matches_ground_truth():
    lm, gt = _build()
    assert lm.axis_direction is not None
    dot = lm.axis_direction[0] * gt["axis_direction"][0] + lm.axis_direction[1] * gt["axis_direction"][1]
    assert dot > 0.999  # within ~2.5 degrees


# --- fail-safe conditions ----------------------------------------------------------

def test_reject_when_no_part_detected():
    lm = extract_shifter_fork_landmarks(None, circles=[])
    assert lm.reject_reason is not None
    assert "no part" in lm.reject_reason.lower()


def test_reject_when_part_touches_border():
    img, _ = make_shifter_fork_part(margin=0)  # forces the silhouette to the image edge
    part = segmentation.build_detected_part(img)
    assert part is not None
    assert part.touches_border
    lm = extract_shifter_fork_landmarks(part, circles=[])
    assert lm.reject_reason is not None
    assert "border" in lm.reject_reason.lower()


def test_reject_when_no_bore_found():
    img, _ = make_shifter_fork_part(bore_radius_px=0)
    part = segmentation.build_detected_part(img)
    assert part is not None
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    assert circles == []
    lm = extract_shifter_fork_landmarks(part, circles)
    assert lm.reject_reason is not None
    assert "bore" in lm.reject_reason.lower()


def test_reject_when_inner_slot_not_found():
    # A plain rectangle has no dominant concavity at all -- find_fork_tips
    # must return None, and that must fail safe, not crash or guess.
    gray = np.full((400, 400), 225, dtype=np.uint8)
    cv2.rectangle(gray, (100, 100), (300, 300), 40, -1)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    part = segmentation.build_detected_part(img)
    assert part is not None
    lm = extract_shifter_fork_landmarks(part, circles=[])
    assert lm.reject_reason is not None
