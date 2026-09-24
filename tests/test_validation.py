import math

import cv2
import numpy as np

from tests.fixtures import make_axis_aligned_part, make_c_shape_part
from vision import geometry, segmentation, validation
from vision.types import DetectedPart


def _make_star_part(inner_r: float, outer_r: float, n_points: int, canvas_hw=(600, 600)) -> DetectedPart:
    """A deliberately pathological star-shaped contour, built directly
    (bypassing the segmentation pipeline) for controlled gate testing — the
    same approach the geometry hole-detection tests use for their gates."""
    center = (canvas_hw[1] // 2, canvas_hw[0] // 2)
    pts = []
    for i in range(n_points * 2):
        r = outer_r if i % 2 == 0 else inner_r
        a = math.pi * i / n_points
        pts.append((int(center[0] + r * math.cos(a)), int(center[1] + r * math.sin(a))))
    contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    mask = np.zeros(canvas_hw, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    area, perimeter = geometry.area_perimeter(contour)
    return DetectedPart(
        contour=contour, mask=mask, segmentation_method="synthetic",
        bbox=geometry.bounding_box(contour), rotated_rect=geometry.rotated_rect(contour),
        area_px2=area, perimeter_px=perimeter, hole_contours=[], touches_border=False,
    )


def test_no_part_is_not_reliable():
    ok, reason = validation.segmentation_is_reliable(None, (400, 400))
    assert not ok
    assert reason


def test_border_touching_part_is_not_reliable():
    img, _ = make_axis_aligned_part(canvas_hw=(400, 400), rect_xywh=(0, 0, 200, 200))
    part = segmentation.build_detected_part(img)
    assert part is not None
    ok, reason = validation.segmentation_is_reliable(part, img.shape[:2])
    assert not ok
    assert "border" in reason.lower() or "frame" in reason.lower()


def test_normal_part_is_reliable():
    img, _ = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200))
    part = segmentation.build_detected_part(img)
    assert part is not None
    ok, reason = validation.segmentation_is_reliable(part, img.shape[:2])
    assert ok
    assert reason == ""


def test_open_c_shape_part_is_reliable_despite_low_bounding_box_fill():
    # This is the exact bug reported against a real forged C/U-bracket
    # component: a legitimate open/concave part has a lot of intentional
    # empty space in its own bounding rectangle. Low extent must NOT reject
    # it (bounding-rectangle fill is informational only, never a gate).
    img, gt = make_c_shape_part()
    part = segmentation.build_detected_part(img)
    assert part is not None

    diag = validation.evaluate_segmentation(part, img.shape[:2])
    assert diag.extent < 0.55, "fixture should genuinely have low bounding-rect fill, or this test proves nothing"
    assert diag.accepted, f"a legitimate open shape must be accepted; got reasons: {diag.reasons}"

    ok, reason = validation.segmentation_is_reliable(part, img.shape[:2])
    assert ok
    assert reason == ""


def test_low_solidity_contour_is_rejected_regardless_of_shape():
    # A deep 10-point star: extent, solidity, AND compactness are all bad —
    # this is what genuine noise/fragmented segmentation looks like, and it
    # must still be rejected even though bounding-box fill is no longer the
    # criterion doing the rejecting.
    part = _make_star_part(inner_r=15, outer_r=200, n_points=10)
    diag = validation.evaluate_segmentation(part, (600, 600))

    assert diag.solidity < validation.MIN_SOLIDITY_FOR_RELIABLE_CONTOUR
    assert not diag.accepted
    assert any("fragmented" in r.lower() or "irregular" in r.lower() for r in diag.reasons)


def test_low_extent_alone_does_not_trigger_any_reason():
    # Direct regression guard: construct a shape with deliberately low
    # extent but healthy solidity/compactness, and confirm no rejection
    # reason ever mentions extent/rectangle/fill.
    img, gt = make_c_shape_part()
    part = segmentation.build_detected_part(img)
    assert part is not None
    diag = validation.evaluate_segmentation(part, img.shape[:2])
    assert diag.accepted
    combined = " ".join(diag.reasons).lower()
    assert "bounding rectangle" not in combined
    assert "fill" not in combined


def test_hard_quality_gate_blocks_extremely_low_resolution():
    img, _ = make_axis_aligned_part(canvas_hw=(120, 160), rect_xywh=(20, 20, 80, 60))
    report = validation.assess_image_quality(img)
    assert report.has_errors()


def test_soft_quality_warning_does_not_block_moderately_low_resolution():
    img, _ = make_axis_aligned_part(canvas_hw=(300, 400), rect_xywh=(60, 60, 200, 150))
    report = validation.assess_image_quality(img)
    # Below the normal MIN_SHORT_SIDE_PX but above the hard floor: a warning,
    # not a block.
    assert not report.has_errors()
    assert any("resolution" in i.message.lower() for i in report.issues)


def test_hard_quality_gate_blocks_extreme_blur():
    gray_flat = np.full((800, 800, 3), 150, dtype=np.uint8)
    report = validation.assess_image_quality(gray_flat)
    assert report.has_errors()
