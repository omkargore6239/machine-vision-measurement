import numpy as np

from tests.fixtures import make_axis_aligned_part
from vision import segmentation, validation


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
