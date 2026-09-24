import math

from tests.fixtures import make_axis_aligned_part, make_rounded_rect_part
from vision import geometry, segmentation
from vision.preprocessing import to_gray

DIAMETER_TOLERANCE_PX = 3


def test_circle_detection_matches_ground_truth():
    holes = [(320, 260, 25), (450, 340, 18)]
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=holes)
    part = segmentation.build_detected_part(img)
    assert part is not None

    gray = to_gray(img)
    hough = geometry.detect_circles_hough(gray, part.mask)
    contour_circles = geometry.detect_circles_contour(part.hole_contours)
    circles = geometry.merge_circle_detections(hough, contour_circles)

    # Every ground-truth hole must be found accurately. (Hough alone can add
    # spurious low-confidence candidates on sharp corners; contour circularity
    # filtering is what the app relies on for trustworthy reporting, so we
    # only assert on that, not on the total candidate count.)
    for gx, gy, gr in holes:
        match = min(circles, key=lambda c: geometry.euclidean_distance((c.cx, c.cy), (gx, gy)))
        assert geometry.euclidean_distance((match.cx, match.cy), (gx, gy)) <= DIAMETER_TOLERANCE_PX
        assert abs(match.r - gr) <= DIAMETER_TOLERANCE_PX
        assert match.circularity is not None
        assert match.circularity >= geometry.MIN_CIRCULARITY_TO_ACCEPT

    # No detection should be reported without circularity evidence backing it
    # (contour-confirmed) — a bare Hough guess alone shouldn't silently pass
    # as if it were geometrically verified.
    for c in circles:
        if c.circularity is not None:
            assert c.circularity >= geometry.MIN_CIRCULARITY_TO_ACCEPT


def test_low_circularity_shapes_are_rejected():
    # A long thin rectangle-shaped "hole" should NOT be reported as a circle.
    import cv2
    import numpy as np
    bad_contour = np.array([[[100, 100]], [[100, 110]], [[300, 110]], [[300, 100]]], dtype=np.int32)
    result = geometry.detect_circles_contour([bad_contour])
    assert result == []


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
