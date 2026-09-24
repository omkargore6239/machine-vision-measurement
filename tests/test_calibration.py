import pytest

from tests.fixtures import make_axis_aligned_part, make_checkerboard_image
from vision import calibration
from vision.types import CONFIDENCE_LOW


def test_known_dimension_calibration_scale_is_correct():
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200))
    profile = calibration.known_dimension_calibrate(img, known_mm=50.0, direction="horizontal")

    assert profile.method == "known_dimension"
    assert profile.detected_pixels == pytest.approx(300, abs=4)
    assert profile.mm_per_pixel == pytest.approx(50.0 / profile.detected_pixels, rel=1e-9)
    assert profile.confidence != CONFIDENCE_LOW


def test_known_dimension_calibration_rejects_zero_known_mm():
    img, _ = make_axis_aligned_part()
    with pytest.raises(ValueError):
        calibration.known_dimension_calibrate(img, known_mm=0.0, direction="horizontal")


def test_known_dimension_calibration_fails_on_blank_image():
    import numpy as np
    blank = np.full((300, 300, 3), 128, dtype="uint8")
    with pytest.raises(ValueError):
        calibration.known_dimension_calibrate(blank, known_mm=50.0, direction="horizontal")


def test_checkerboard_calibration_scale_only():
    square_px = 40
    img = make_checkerboard_image(square_px=square_px, cols_inner=7, rows_inner=5)
    profile = calibration.checkerboard_calibrate([img], pattern_cols=7, pattern_rows=5, square_size_mm=25.0)

    assert profile.method == "checkerboard"
    expected_mm_per_px = 25.0 / square_px
    assert profile.mm_per_pixel == pytest.approx(expected_mm_per_px, rel=0.05)
    assert profile.camera_matrix is None  # only one image supplied


def test_checkerboard_calibration_with_multiple_images_estimates_camera_matrix():
    square_px = 40
    imgs = [
        make_checkerboard_image(square_px=square_px, cols_inner=7, rows_inner=5, margin=m)
        for m in (40, 70, 100)
    ]
    profile = calibration.checkerboard_calibrate(imgs, pattern_cols=7, pattern_rows=5, square_size_mm=25.0)

    assert profile.camera_matrix is not None
    assert profile.dist_coeffs is not None
    assert profile.reprojection_error_px is not None
    assert profile.checkerboard_images_used == 3


def test_checkerboard_calibration_raises_when_pattern_not_found():
    import numpy as np
    blank = np.full((300, 300, 3), 255, dtype="uint8")
    with pytest.raises(ValueError):
        calibration.checkerboard_calibrate([blank], pattern_cols=7, pattern_rows=5, square_size_mm=25.0)


def test_perspective_calibration_produces_exact_scale():
    import numpy as np
    img = np.full((400, 600, 3), 128, dtype="uint8")
    src_points = [(50, 50), (550, 50), (550, 350), (50, 350)]
    profile, warped = calibration.perspective_calibrate(
        img, src_points, known_width_mm=100.0, known_height_mm=60.0, px_per_mm_target=10.0,
    )

    assert profile.method == "perspective"
    assert profile.mm_per_pixel == pytest.approx(0.1, rel=1e-9)
    assert profile.warped_size == (1000, 600)
    assert warped.shape[1] == 1000
    assert warped.shape[0] == 600


def test_apply_calibration_returns_none_without_profile():
    from vision.calibration import apply_calibration
    assert apply_calibration(100.0, None) is None


def test_save_load_delete_profile_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "CALIBRATION_DIR", tmp_path)

    img, _ = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200))
    profile = calibration.known_dimension_calibrate(img, known_mm=50.0, direction="horizontal", name="test-profile")

    calibration.save_profile(profile)
    loaded = calibration.load_profile(profile.profile_id)
    assert loaded.name == "test-profile"
    assert loaded.mm_per_pixel == pytest.approx(profile.mm_per_pixel)

    profiles = calibration.list_profiles()
    assert any(p.profile_id == profile.profile_id for p in profiles)

    calibration.delete_profile(profile.profile_id)
    assert not any(p.profile_id == profile.profile_id for p in calibration.list_profiles())
