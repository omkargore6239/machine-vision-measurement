from tests.fixtures import make_axis_aligned_part
from vision import segmentation

BBOX_TOLERANCE_PX = 4


def test_detects_rectangle_bbox():
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=None)
    part = segmentation.build_detected_part(img)

    assert part is not None
    x, y, w, h = part.bbox
    gx, gy, gw, gh = gt["bbox"]
    assert abs(x - gx) <= BBOX_TOLERANCE_PX
    assert abs(y - gy) <= BBOX_TOLERANCE_PX
    assert abs(w - gw) <= BBOX_TOLERANCE_PX
    assert abs(h - gh) <= BBOX_TOLERANCE_PX
    assert not part.touches_border


def test_detects_holes():
    holes = [(320, 260, 20), (450, 340, 20)]
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=holes)
    part = segmentation.build_detected_part(img)

    assert part is not None
    assert len(part.hole_contours) == len(holes)


def test_no_part_in_blank_image():
    import numpy as np
    blank = np.full((400, 400, 3), 128, dtype="uint8")
    part = segmentation.build_detected_part(blank)
    assert part is None


def test_part_touching_border_is_flagged():
    img, _ = make_axis_aligned_part(canvas_hw=(400, 400), rect_xywh=(0, 0, 200, 200))
    part = segmentation.build_detected_part(img)
    assert part is not None
    assert part.touches_border
