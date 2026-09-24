import math

import cv2
import numpy as np

from tests import fixtures
from tests.fixtures import make_axis_aligned_part, make_rounded_rect_part
from vision import geometry, segmentation
from vision.preprocessing import to_gray

DIAMETER_TOLERANCE_PX = 4


def _detect(img):
    part = segmentation.build_detected_part(img)
    assert part is not None
    gray = to_gray(img)
    circles, candidates = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2)
    return part, circles, candidates


def test_circle_detection_matches_ground_truth():
    holes = [(320, 260, 25), (450, 340, 18)]
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=holes)
    part, circles, candidates = _detect(img)

    assert len(circles) == len(holes)
    for gx, gy, gr in holes:
        match = min(circles, key=lambda c: geometry.euclidean_distance((c.cx, c.cy), (gx, gy)))
        assert geometry.euclidean_distance((match.cx, match.cy), (gx, gy)) <= DIAMETER_TOLERANCE_PX
        assert abs(match.r - gr) <= DIAMETER_TOLERANCE_PX
        assert match.circularity is not None
        assert match.circularity >= geometry.MIN_CIRCULARITY


def test_zero_holes_when_none_present():
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=None)
    part, circles, candidates = _detect(img)
    assert circles == []


def test_single_real_hole_detected_exactly_once():
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=[(400, 300, 22)])
    part, circles, candidates = _detect(img)
    assert len(circles) == 1


def test_four_real_holes_detected_exactly():
    holes = [(320, 260, 20), (480, 260, 18), (320, 340, 22), (480, 340, 16)]
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=holes)
    part, circles, candidates = _detect(img)
    assert len(circles) == 4


def test_hough_and_contour_agreement_yields_one_measurement_end_to_end():
    # Full pipeline (not the unit-level evaluate_hole_candidates test above):
    # a real hole that both Hough and the contour pass independently confirm
    # must still show up as exactly one CircleFeature.
    holes = [(400, 300, 24)]
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=holes)
    part, circles, candidates = _detect(img)
    assert len(circles) == 1
    assert circles[0].method == "hough+contour"
    assert circles[0].hough_confirmed


def test_text_alone_does_not_produce_false_holes():
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 450, 250), holes=None)
    img = fixtures.add_printed_text(img, gt["bbox"], text="MODEL 8080 QD@")
    part, circles, candidates = _detect(img)
    assert circles == []


def test_texture_noise_alone_does_not_produce_false_holes():
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 450, 250), holes=None)
    img = fixtures.add_texture_noise(img, gt["bbox"], n_specks=300)
    part, circles, candidates = _detect(img)
    assert circles == []


def test_real_holes_survive_alongside_text_and_texture():
    holes = [(320, 320, 22), (420, 360, 18), (520, 320, 20), (620, 360, 16)]
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 450, 250), holes=holes)
    img = fixtures.add_printed_text(img, gt["bbox"], text="MODEL 8080 QD@")
    img = fixtures.add_texture_noise(img, gt["bbox"], n_specks=300, avoid_regions=holes)

    part, circles, candidates = _detect(img)

    assert len(circles) == len(holes)
    for gx, gy, gr in holes:
        match = min(circles, key=lambda c: geometry.euclidean_distance((c.cx, c.cy), (gx, gy)))
        assert geometry.euclidean_distance((match.cx, match.cy), (gx, gy)) <= DIAMETER_TOLERANCE_PX


def test_hough_alone_never_reports_a_hole():
    # A Hough hit with no corresponding contour candidate must never become
    # a reported CircleFeature — this is the core fix for the false-positive
    # flood (previously, an unmatched Hough hit was reported at MEDIUM
    # confidence on its own).
    hough_only = [{"cx": 150.0, "cy": 150.0, "r": 20.0}]
    candidates = geometry.evaluate_hole_candidates(
        gray=np.full((300, 300), 200, dtype=np.uint8),
        outer_mask=np.full((300, 300), 255, dtype=np.uint8),
        hole_contours=[],  # no contour-confirmed candidates at all
        hough_hits=hough_only,
        part_area_px2=300 * 300,
    )
    assert candidates == []
    assert geometry.hole_candidates_to_circle_features(candidates) == []


def test_moderately_elongated_ellipse_rejected_by_aspect_ratio_gate():
    # circularity alone would accept this (0.88 for a 2.5:1 ellipse can still
    # clear MIN_CIRCULARITY) — the aspect-ratio gate is what correctly
    # rejects an oval this elongated from being called a "hole".
    canvas = np.zeros((300, 300), dtype=np.uint8)
    cv2.ellipse(canvas, (150, 150), (50, 20), 0, 0, 360, 255, -1)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipse_contour = max(contours, key=cv2.contourArea)

    gray = np.full((300, 300), 200, dtype=np.uint8)
    gray[canvas == 255] = 60
    outer_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(outer_mask, (10, 10), (290, 290), 255, -1)

    candidates = geometry.evaluate_hole_candidates(gray, outer_mask, [ellipse_contour], [], part_area_px2=280 * 280)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.aspect_ratio > geometry.MAX_ASPECT_RATIO
    assert not c.accepted
    assert any("elongated" in r.lower() for r in c.rejection_reasons)


def test_weak_contrast_candidate_is_rejected():
    canvas = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 25, 255, -1)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    circle_contour = max(contours, key=cv2.contourArea)

    gray = np.full((300, 300), 200, dtype=np.uint8)
    gray[canvas == 255] = 195  # barely different from surrounding material
    outer_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(outer_mask, (10, 10), (290, 290), 255, -1)

    candidates = geometry.evaluate_hole_candidates(gray, outer_mask, [circle_contour], [], part_area_px2=280 * 280)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.contrast < geometry.MIN_CONTRAST
    assert not c.accepted
    assert any("contrast" in r.lower() for r in c.rejection_reasons)


def test_too_small_candidate_is_rejected():
    canvas = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 2, 255, -1)  # ~4px diameter
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tiny_contour = max(contours, key=cv2.contourArea)

    gray = np.full((300, 300), 200, dtype=np.uint8)
    gray[canvas == 255] = 40
    outer_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(outer_mask, (10, 10), (290, 290), 255, -1)

    candidates = geometry.evaluate_hole_candidates(gray, outer_mask, [tiny_contour], [], part_area_px2=280 * 280)
    assert len(candidates) == 1
    assert not candidates[0].accepted


def test_hough_confirmation_boosts_confidence():
    canvas = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 25, 255, -1)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    circle_contour = max(contours, key=cv2.contourArea)

    gray = np.full((300, 300), 200, dtype=np.uint8)
    gray[canvas == 255] = 40
    outer_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(outer_mask, (10, 10), (290, 290), 255, -1)

    without_hough = geometry.evaluate_hole_candidates(gray, outer_mask, [circle_contour], [], part_area_px2=280 * 280)
    with_hough = geometry.evaluate_hole_candidates(
        gray, outer_mask, [circle_contour], [{"cx": 150.0, "cy": 150.0, "r": 25.0}], part_area_px2=280 * 280,
    )

    assert without_hough[0].accepted and with_hough[0].accepted
    assert with_hough[0].hough_confirmed and not without_hough[0].hough_confirmed
    assert with_hough[0].confidence_score >= without_hough[0].confidence_score


def test_duplicate_candidates_are_merged_to_one():
    # Two near-identical overlapping circles at the same location should
    # never produce two separate reported measurements for one physical hole.
    canvas1 = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(canvas1, (150, 150), 25, 255, -1)
    canvas2 = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(canvas2, (152, 149), 25, 255, -1)  # 2-3px offset, same physical hole

    c1 = max(cv2.findContours(canvas1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    c2 = max(cv2.findContours(canvas2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)

    gray = np.full((300, 300), 200, dtype=np.uint8)
    gray[canvas1 == 255] = 40
    gray[canvas2 == 255] = 40
    outer_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(outer_mask, (10, 10), (290, 290), 255, -1)

    candidates = geometry.evaluate_hole_candidates(gray, outer_mask, [c1, c2], [], part_area_px2=280 * 280)
    features = geometry.hole_candidates_to_circle_features(candidates)
    assert len(features) == 1


def test_euclidean_distance():
    assert geometry.euclidean_distance((0, 0), (3, 4)) == 5.0


def test_angles_of_axis_aligned_rectangle_are_90_degrees():
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200))
    part = segmentation.build_detected_part(img)
    assert part is not None

    angles = geometry.estimate_angles(part.contour)
    reliable = [a for a in angles if a["reliable"]]
    assert len(reliable) == 4
    for a in reliable:
        assert math.isclose(a["angle_deg"], 90.0, abs_tol=3.0)


def test_corner_radius_detects_rounded_corner_reasonably():
    img, gt = make_rounded_rect_part(rect_xywh=(250, 200, 300, 200), corner_radius=30)
    part = segmentation.build_detected_part(img)
    assert part is not None

    corners = geometry.estimate_corner_radii(part.contour)
    reliable = [c for c in corners if c["reliable"]]
    assert len(reliable) >= 1
    for c in reliable:
        assert abs(c["radius_px"] - gt["corner_radius"]) <= 0.4 * gt["corner_radius"]


def test_rotated_rect_normalizes_width_height():
    img, _ = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200))
    part = segmentation.build_detected_part(img)
    _, _, rw, rh, _ = part.rotated_rect
    assert rw >= rh
