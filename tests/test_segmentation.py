import cv2
import numpy as np
import pytest

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


def test_nested_candidate_inside_a_boundary_ring_is_not_hidden():
    # Regression for a real forged part photo: a genuine hole that sits
    # geometrically inside a thin boundary-bleed "ring" artifact (from mask
    # over-inclusion) is topologically NESTED in that ring's contour
    # hierarchy even though it's a completely separate, disconnected blob.
    # Plain RETR_EXTERNAL silently drops anything not at hierarchy depth 0,
    # hiding it entirely — confirmed as the exact mechanism that made a real,
    # large, obvious bore vanish. `_find_contours_excluding_hole_traces`
    # (RETR_CCOMP + keeping only parent==-1 entries) must recover it.
    mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.rectangle(mask, (136, 136), (663, 563), 255, thickness=8)  # thin ring artifact
    cv2.circle(mask, (320, 260), 15, 255, -1)  # separate dot, nested inside the ring's hole

    old_way, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(old_way) == 1, "test setup sanity check: RETR_EXTERNAL should hide the dot here"

    found = segmentation._find_contours_excluding_hole_traces(mask)
    areas = sorted(cv2.contourArea(c) for c in found)
    assert len(areas) == 2
    assert areas[0] == pytest.approx(706.0, abs=50), "the recovered small dot"
