import math

import cv2
import numpy as np

from tests import fixtures
from tests.fixtures import (
    make_axis_aligned_part, make_circular_flange_part, make_fork_bracket_part, make_rounded_rect_part,
)
from vision import geometry, measurement, segmentation
from vision.preprocessing import to_gray

DIAMETER_TOLERANCE_PX = 4


def _detect(img):
    part = segmentation.build_detected_part(img)
    assert part is not None
    gray = to_gray(img)
    circles, candidates = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
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


def _make_scalloped_hole_part(scallop_amplitude_px=8, n_scallops=14, hole_radius=70):
    """A rectangular part with one hole whose boundary is a scalloped/
    jagged circle rather than a smooth one — mimicking the exact failure
    mode found on a real photographed bore (a genuinely round, high-
    solidity hole whose raw circularity read 0.63-0.68, well below
    MIN_CIRCULARITY, purely from adaptive-threshold/rim-texture boundary
    noise — see `geometry._smooth_contour_for_shape_metrics`'s docstring)."""
    canvas_hw = (600, 800)
    rect_xywh = (250, 200, 300, 200)
    gray = np.full(canvas_hw, 40, dtype=np.uint8)
    x, y, w, h = rect_xywh
    gray[y:y + h, x:x + w] = 210
    cx, cy = x + w // 2, y + h // 2
    n_pts = n_scallops * 6
    pts = []
    for i in range(n_pts):
        angle = 2 * math.pi * i / n_pts
        bump = scallop_amplitude_px if (i // 3) % 2 == 0 else 0
        r = hole_radius + bump
        pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
    cv2.fillPoly(gray, [np.array(pts, dtype=np.int32)], 40)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img, {"center": (cx, cy), "radius": hole_radius}


def test_scalloped_hole_boundary_is_still_detected_after_smoothing():
    # A genuinely round hole with small-amplitude boundary scalloping must
    # still be detected as a hole -- this is exactly the failure mode
    # `_smooth_contour_for_shape_metrics` was added to fix.
    img, gt = _make_scalloped_hole_part(scallop_amplitude_px=8, n_scallops=14, hole_radius=70)
    part, circles, candidates = _detect(img)
    assert len(circles) == 1
    c = circles[0]
    assert geometry.euclidean_distance((c.cx, c.cy), gt["center"]) <= DIAMETER_TOLERANCE_PX * 2
    assert abs(c.r - gt["radius"]) <= DIAMETER_TOLERANCE_PX * 3


def test_genuinely_non_circular_shape_still_rejected_after_smoothing():
    # A real star/gear shape (large-amplitude, coarse scalloping -- a
    # genuinely different shape, not fine boundary noise) must still fail —
    # smoothing must not make every jagged blob pass.
    img, gt = _make_scalloped_hole_part(scallop_amplitude_px=45, n_scallops=6, hole_radius=60)
    part, circles, candidates = _detect(img)
    assert len(circles) == 0


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


# --- internal-feature detection (circular flange / central bore) ------------------

def test_central_bore_alone_is_detected_and_classified():
    # Spec item D: a large central bore with NO other holes must still be
    # found — it must not get lost to the "too large relative to part" gate
    # the way it previously did in segmentation's old fixed 50% area cap.
    img, gt = make_circular_flange_part(n_bolt_holes=0, center_bore_r=90)
    part, circles, candidates = _detect(img)
    circles = measurement.classify_and_label_circles(circles, part)

    assert len(circles) == 1
    assert circles[0].feature_type == "bore"
    assert circles[0].label == "Center Bore"
    assert abs(2 * circles[0].r - 180) <= DIAMETER_TOLERANCE_PX * 2


def test_multiple_holes_plus_central_bore_all_detected_and_labeled():
    # Spec item E: this is the exact reported failure scenario (a flange with
    # several bolt holes AND a center bore) — all of them must be found, and
    # the bore must be distinguishable from the regular holes.
    img, gt = make_circular_flange_part()
    part, circles, candidates = _detect(img)
    circles = measurement.classify_and_label_circles(circles, part)

    assert len(circles) == len(gt["holes"]) == 7
    bores = [c for c in circles if c.feature_type == "bore"]
    holes = [c for c in circles if c.feature_type == "hole"]
    assert len(bores) == 1
    assert len(holes) == 6
    assert {c.label for c in holes} == {f"Hole {i}" for i in range(1, 7)}


def test_decorative_circular_ring_is_not_a_hole():
    # Spec item G: a printed/embossed circular decoration (a ring outline,
    # not a hole) — hollow center reads as material color, so this should
    # fail the interior-consistency check the same way ring-shaped text does.
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 400, 300), holes=None)
    x, y, w, h = gt["bbox"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.circle(gray, (x + w // 2, y + h // 2), 40, 150, 4)  # thin printed ring outline
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    part, circles, candidates = _detect(img)
    assert circles == []


def test_specular_reflection_is_not_a_hole():
    # Spec item H: a soft, gradient-falloff bright blob (a specular
    # highlight) should be rejected — no crisp edge, no uniform interior.
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 400, 300), holes=None)
    x, y, w, h = gt["bbox"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    yy, xx = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    cx, cy, radius = x + w // 2, y + h // 2, 35
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    highlight = np.clip(60 * (1 - dist / radius), 0, None)
    gray = np.clip(gray + highlight, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    part, circles, candidates = _detect(img)
    assert circles == []


def test_moderate_perspective_ellipse_is_accepted_with_major_minor():
    # Spec item L: a real circular hole photographed at an angle appears as
    # a moderately-elongated ellipse — this must still be accepted (unlike
    # the more extreme 2.5:1 case tested elsewhere), with major/minor
    # diameters both reported. Not a claim of measuring 3D depth — just that
    # a plausible perspective-skewed circle isn't thrown out as "not round".
    canvas = np.zeros((300, 300), dtype=np.uint8)
    cv2.ellipse(canvas, (150, 150), (40, 27), 15, 0, 360, 255, -1)  # aspect ~1.48
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipse_contour = max(contours, key=cv2.contourArea)

    gray = np.full((300, 300), 200, dtype=np.uint8)
    gray[canvas == 255] = 40
    outer_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(outer_mask, (10, 10), (290, 290), 255, -1)

    candidates = geometry.evaluate_hole_candidates(gray, outer_mask, [ellipse_contour], [], part_area_px2=280 * 280)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.accepted
    assert c.aspect_ratio <= geometry.MAX_ASPECT_RATIO
    assert c.major_axis_px is not None and c.minor_axis_px is not None
    assert c.major_axis_px > c.minor_axis_px


def test_candidate_too_close_to_part_edge_is_rejected_using_true_contour():
    # The position gate uses distance to the part's TRUE outer boundary, not
    # its bounding-box edges — important for circular/irregular parts, where
    # most of the bbox near its corners isn't part material at all.
    canvas = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(canvas, (200, 200), 150, 255, -1)  # circular part
    outer_contour = max(cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)

    # A "hole" candidate right at the part's true edge (top of the circle,
    # far from the bbox corners, but only ~3px inside the true boundary).
    hole_canvas = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(hole_canvas, (200, 53), 10, 255, -1)
    hole_contour = max(cv2.findContours(hole_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)

    gray = np.full((400, 400), 200, dtype=np.uint8)
    gray[hole_canvas == 255] = 40
    outer_mask = canvas

    candidates = geometry.evaluate_hole_candidates(
        gray, outer_mask, [hole_contour], [], part_area_px2=cv2.countNonZero(canvas), outer_contour=outer_contour,
    )
    assert len(candidates) == 1
    assert not candidates[0].accepted
    assert any("edge" in r.lower() for r in candidates[0].rejection_reasons)


def test_candidate_larger_than_area_gate_is_rejected():
    canvas = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 130, 255, -1)  # ~68% of a 280x280 part area
    contour = max(cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)

    gray = np.full((300, 300), 200, dtype=np.uint8)
    gray[canvas == 255] = 40
    outer_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(outer_mask, (10, 10), (290, 290), 255, -1)

    part_area = 280 * 280
    assert (cv2.contourArea(contour) / part_area) > geometry.MAX_AREA_FRACTION_OF_PART

    candidates = geometry.evaluate_hole_candidates(gray, outer_mask, [contour], [], part_area_px2=part_area)
    assert len(candidates) == 1
    assert not candidates[0].accepted
    assert any("large" in r.lower() for r in candidates[0].rejection_reasons)


def test_multi_method_candidate_generation_still_deduplicates_to_one_per_hole():
    # Spec item I (extended): with three independent candidate-generation
    # methods now feeding evaluate_hole_candidates (see segmentation.
    # extract_hole_contours), a real hole must still resolve to exactly one
    # measurement, not up to three near-duplicates.
    holes = [(400, 300, 24)]
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=holes)
    part = segmentation.build_detected_part(img)
    raw_contours = part.hole_contours
    assert len(raw_contours) == 1, "raw candidate generation should already dedupe near-identical proposals"

    part2, circles, candidates = _detect(img)
    assert len(circles) == 1


def test_hole_with_internal_specular_highlight_is_still_detected():
    # Regression for a real forged/polished-metal part: a bright reflection
    # streak crossing through a genuine hole's dark interior used to
    # fragment it into two non-circular crescents (each independently
    # failing circularity) instead of being recognized as one hole with a
    # highlight. The fix is dark_majority_fraction in
    # `_local_contrast_and_uniformity`, which allows a high interior_std
    # through when the interior is still MAJORITY dark (unlike a hollow
    # ring/stroke, which is majority background).
    holes = [(400, 300, 60)]
    img, gt = make_axis_aligned_part(rect_xywh=(200, 150, 400, 350), holes=holes)
    # Paint a bright diagonal highlight stripe across the hole's interior.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.line(gray, (370, 270), (430, 330), 230, thickness=14)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    part, circles, candidates = _detect(img)
    assert len(circles) == 1
    match = circles[0]
    assert geometry.euclidean_distance((match.cx, match.cy), (400, 300)) <= DIAMETER_TOLERANCE_PX * 2


def test_hole_near_image_border_is_still_detected_when_refinement_is_skipped():
    # Regression for a real forged part photo: when the part sits close
    # enough to the image edge that segmentation's local-boundary-refinement
    # crop would be clamped, refinement is skipped (see
    # `segmentation._refine_contour_locally`) and the coarser, unrefined
    # outer contour is used instead. That coarser contour can bleed a few
    # pixels past the true edge, creating a thin boundary artifact that used
    # to silently hide any real hole sitting inside it (see
    # `segmentation._find_contours_excluding_hole_traces`). This must not
    # happen: a hole comfortably inside the part must still be found even
    # when the part itself is flush against the frame.
    holes = [(320, 260, 15)]
    img, gt = make_axis_aligned_part(canvas_hw=(600, 800), rect_xywh=(150, 150, 500, 400), holes=holes)
    part, circles, candidates = _detect(img)
    assert len(circles) == 1
    assert geometry.euclidean_distance((circles[0].cx, circles[0].cy), (320, 260)) <= DIAMETER_TOLERANCE_PX * 2


def test_false_hough_circles_on_busy_flange_do_not_inflate_count():
    # Spec item N, on a circular (not rectangular) part: printed text placed
    # in the material gap between bolt holes must not add spurious detections.
    img, gt = make_circular_flange_part()  # default: 6 bolt holes at 60-degree spacing + bore
    cx, cy = gt["center"]
    # 30 degrees sits exactly between the 0- and 60-degree bolt holes; radius
    # 110 sits well between the center bore (r=40) and the bolt circle (r=180).
    tx = cx + 110 * math.cos(math.radians(30)) - 45
    ty = cy + 110 * math.sin(math.radians(30)) - 15
    img = fixtures.add_printed_text(img, (int(tx), int(ty), 90, 30), text="8080", ink_offset=-70, font_scale=0.8)

    part, circles, candidates = _detect(img)
    assert len(circles) == len(gt["holes"])


# --- fork/opening geometry: convex-hull defect analysis -----------------------------

FORK_TOLERANCE_PX = 12  # pixelization/discretization tolerance for corner-precision checks


def _build_fork(**kwargs):
    img, gt = make_fork_bracket_part(**kwargs)
    part = segmentation.build_detected_part(img)
    assert part is not None
    return img, gt, part


def test_find_fork_tips_locates_both_inner_corners():
    img, gt, part = _build_fork()
    tips = geometry.find_fork_tips(part.contour, part.area_px2)
    assert tips is not None

    found = sorted([tips["tip_a"], tips["tip_b"]], key=lambda p: p[0])
    expected = sorted([gt["tip_a_inner"], gt["tip_b_inner"]], key=lambda p: p[0])
    for f, e in zip(found, expected):
        assert geometry.euclidean_distance(f, e) <= FORK_TOLERANCE_PX


def test_find_fork_tips_none_on_a_plain_circle():
    # A plain filled circle has no dominant concavity at all -- must not
    # fabricate "tips" from contour noise.
    img, gt = make_circular_flange_part(n_bolt_holes=0, center_bore_r=0)
    part = segmentation.build_detected_part(img)
    assert part is not None
    tips = geometry.find_fork_tips(part.contour, part.area_px2)
    assert tips is None


def test_fit_concave_arc_matches_inner_radius_despite_hub_bump():
    img, gt, part = _build_fork()
    tips = geometry.find_fork_tips(part.contour, part.area_px2)
    assert tips is not None
    arc = geometry.fit_concave_arc(part.contour, tips["start_idx"], tips["end_idx"])
    assert arc is not None
    assert abs(arc["radius_px"] - gt["inner_radius"]) <= FORK_TOLERANCE_PX * 2
    assert arc["angular_span_deg"] >= geometry.MIN_ARC_ANGULAR_SPAN_DEG
    # the hub bump is a real, deliberate outlier in the run -- a robust fit
    # must not need every single point to agree
    assert arc["inlier_fraction"] < 1.0


def test_fit_concave_arc_none_without_a_real_concave_run():
    # Too few points in the run (a tiny defect) must not produce a fitted
    # "diameter" from noise.
    tiny_contour = np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]], dtype=np.int32)
    assert geometry.fit_concave_arc(tiny_contour, 0, 2) is None


def test_measure_tip_thickness_matches_arm_thickness():
    img, gt, part = _build_fork()
    tips = geometry.find_fork_tips(part.contour, part.area_px2)
    arc = geometry.fit_concave_arc(part.contour, tips["start_idx"], tips["end_idx"])
    assert arc is not None
    ref_center = (arc["cx"], arc["cy"])
    run = geometry._contour_run(part.contour, tips["start_idx"], tips["end_idx"])
    inset = 15
    sample_a = tuple(run[inset])
    sample_b = tuple(run[-inset - 1])

    t_a = geometry.measure_tip_thickness(part.mask, sample_a, ref_center)
    t_b = geometry.measure_tip_thickness(part.mask, sample_b, ref_center)
    assert t_a is not None and t_b is not None
    assert abs(t_a["thickness_px"] - gt["arm_thickness_px"]) <= 5
    assert abs(t_b["thickness_px"] - gt["arm_thickness_px"]) <= 5


def test_measure_tip_thickness_none_for_degenerate_direction():
    mask = np.full((50, 50), 255, dtype=np.uint8)
    assert geometry.measure_tip_thickness(mask, (10, 10), (10, 10)) is None  # zero-length direction


def test_fit_convex_arc_matches_outer_radius_and_exceeds_inner():
    img, gt, part = _build_fork()
    tips = geometry.find_fork_tips(part.contour, part.area_px2)
    assert tips is not None
    inner = geometry.fit_concave_arc(part.contour, tips["start_idx"], tips["end_idx"])
    outer = geometry.fit_convex_arc(part.contour, tips["start_idx"], tips["end_idx"])
    assert outer is not None
    assert abs(outer["radius_px"] - gt["outer_radius"]) <= FORK_TOLERANCE_PX * 3
    assert outer["angular_span_deg"] >= geometry.MIN_ARC_ANGULAR_SPAN_DEG
    # sanity: outer must be a genuinely larger radius than the inner fit,
    # not the same fit accidentally reused for both sides
    assert inner is not None
    assert outer["radius_px"] > inner["radius_px"]
