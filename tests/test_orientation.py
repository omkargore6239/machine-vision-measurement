"""Rotation robustness for the inspection-spec measurements. A real part
photographed at a different angle (0/90/180/270/arbitrary) must produce the
SAME measurements, not different ones -- these tests are the actual proof
of that, not just a unit test of a formula in isolation.

Every diameter/radius in the spec comes from a fitted circle, and every
distance (Fork Tip Gap, Fork Arm Length, Hole Center-to-Center, Fork Tip
Thickness) comes from Euclidean distance between two detected points --
both are rotation-invariant by construction, with no code change needed.
The two parameters that AREN'T (Overall Fork Width / Overall Height) were
previously sourced from `cv2.boundingRect` ("Bounding Box Width/Height"),
which is axis-aligned to the *image* and swaps on a 90-degree rotation;
`vision/measurement.py` now sources them from `part.rotated_rect`
("Rotated Rect Width/Height", `cv2.minAreaRect`) instead, which is tied to
the part's own shape and does not swap.

No real BMP dataset was available to this session, so this uses the
synthetic fork-bracket fixture (`tests/fixtures.make_fork_bracket_part`),
rotated programmatically with `cv2.warpAffine` around the image center with
an enlarged canvas (nothing cropped, nothing rescaled) -- a pure rotation,
which is why any measurement drift here would be a real bug, not a
side-effect of resampling.
"""
import cv2
import numpy as np
import pytest

from tests.fixtures import make_fork_bracket_part
from vision import calibration, geometry, measurement, segmentation
from vision.inspection_spec import INSPECTION_SPEC
from vision.preprocessing import to_gray

# Parameters this synthetic fixture actually models geometrically AND that
# are rotation-invariant by construction (fitted-circle diameter/radius, or
# Euclidean distance between two detected points). Excludes: parameters
# with `not_detectable_reason` (the fixture was never meant to exercise
# those -- they're NOT DETECTED regardless of rotation), `part_thickness`
# (requires_second_view -- never measurable from one 2D photo), and
# `overall_fork_width`/`overall_height` (axis-dependent, checked in their
# own test below).
ROTATION_INVARIANT_IDS = {
    s.parameter_id for s in INSPECTION_SPEC
    if s.parameter_id not in {"overall_fork_width", "overall_height"}
    and s.not_detectable_reason is None
    and not s.requires_second_view
}


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += (new_w - w) / 2.0
    m[1, 2] += (new_h - h) / 2.0
    bg = int(img[0, 0, 0])
    return cv2.warpAffine(img, m, (new_w, new_h), borderValue=(bg, bg, bg))


def _measure(img, profile):
    part = segmentation.build_detected_part(img)
    assert part is not None
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    circles = measurement.classify_and_label_circles(circles, part)
    x, y, w, h = part.bbox
    records = measurement.build_measurement_records(part, circles, profile, (float(x), float(y)))
    results = {
        r.parameter_id: r
        for r in measurement.evaluate_inspection_spec(part, circles, records, profile)
    }
    return part, results


@pytest.fixture(scope="module")
def base_and_profile():
    img, _ = make_fork_bracket_part()
    profile = calibration.known_dimension_calibrate(img, known_mm=50.0, direction="horizontal")
    return img, profile


@pytest.mark.parametrize("angle_deg", [0.0, 90.0, 180.0, 270.0, 37.0])
def test_inspection_spec_measurements_are_rotation_invariant(base_and_profile, angle_deg):
    base_img, profile = base_and_profile
    _, base_results = _measure(base_img, profile)

    rotated_img = _rotate_image(base_img, angle_deg)
    _, results = _measure(rotated_img, profile)

    # 0/90/180/270 are exact pixel remaps (no resampling blur), so those get
    # a tight tolerance. An arbitrary angle (37) genuinely resamples every
    # pixel through bilinear interpolation, which blurs the boundary the
    # fork-tip convexity-defect corner picker and hole-edge fits key off of
    # -- a small amount of corner-localization drift there is real
    # image-resampling noise, not a measurement bug, so it gets a looser
    # (but still tight relative to the ~37-105mm feature sizes) tolerance.
    tol_mm = 0.5 if angle_deg % 90 == 0 else 2.0
    for pid in ROTATION_INVARIANT_IDS:
        base_val = base_results[pid].measured_mm
        if base_val is None:
            continue  # not detected even at 0 degrees -- nothing to compare
        rot_val = results[pid].measured_mm
        assert rot_val is not None, f"{pid}: detected at 0 deg but not at {angle_deg} deg"
        assert abs(rot_val - base_val) < tol_mm, (
            f"{pid}: {base_val:.3f}mm at 0 deg vs {rot_val:.3f}mm at {angle_deg} deg"
        )


@pytest.mark.parametrize("angle_deg", [90.0, 180.0, 270.0])
def test_overall_width_and_height_do_not_swap_on_rotation(base_and_profile, angle_deg):
    # This is the regression this fix exists for: before switching to
    # Rotated Rect Width/Height, a 90-degree-rotated photo of the same part
    # would swap Overall Fork Width and Overall Height (cv2.boundingRect is
    # axis-aligned to the image, not the part).
    base_img, profile = base_and_profile
    _, base_results = _measure(base_img, profile)

    rotated_img = _rotate_image(base_img, angle_deg)
    _, results = _measure(rotated_img, profile)

    for pid in ("overall_fork_width", "overall_height"):
        base_val = base_results[pid].measured_mm
        rot_val = results[pid].measured_mm
        assert base_val is not None and rot_val is not None
        assert abs(rot_val - base_val) < 1.0, (
            f"{pid}: {base_val:.3f}mm at 0 deg vs {rot_val:.3f}mm at {angle_deg} deg"
        )


def test_estimate_part_orientation_reports_a_direction_when_confident():
    img, _ = make_fork_bracket_part()
    part = segmentation.build_detected_part(img)
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    circles = measurement.classify_and_label_circles(circles, part)
    tips = geometry.find_fork_tips(part.contour, part.area_px2)

    orientation = geometry.estimate_part_orientation(part, circles, tips)
    assert orientation["confidence"] == "HIGH"
    assert orientation["angle_deg"] is not None
    assert 0.0 <= orientation["angle_deg"] < 360.0
    assert orientation["reference_points"]  # real points recorded for debug viz


def test_estimate_part_orientation_never_guesses_when_tips_missing():
    img, _ = make_fork_bracket_part()
    part = segmentation.build_detected_part(img)
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    circles = measurement.classify_and_label_circles(circles, part)

    orientation = geometry.estimate_part_orientation(part, circles, tips=None)
    assert orientation["angle_deg"] is None
    assert orientation["confidence"] == "LOW"
    assert "ambiguous" in orientation["reason"].lower()
    # rect_angle_deg needs no other feature, so it's still reported
    assert orientation["rect_angle_deg"] is not None


def test_estimate_part_orientation_never_guesses_when_no_holes():
    img, _ = make_fork_bracket_part()
    part = segmentation.build_detected_part(img)
    tips = geometry.find_fork_tips(part.contour, part.area_px2)

    orientation = geometry.estimate_part_orientation(part, circles=[], tips=tips)
    assert orientation["angle_deg"] is None
    assert orientation["confidence"] == "LOW"
    assert "main hole" in orientation["reason"]
