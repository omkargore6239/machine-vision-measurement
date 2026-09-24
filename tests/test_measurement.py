import pytest

from tests.fixtures import make_axis_aligned_part
from vision import calibration, geometry, measurement, segmentation
from vision.preprocessing import to_gray
from vision.types import MM_UNAVAILABLE_STATUS, ToleranceSpec


def _build_part_and_circles():
    holes = [(320, 260, 20)]
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=holes)
    part = segmentation.build_detected_part(img)
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2)
    return part, circles


def test_no_calibration_never_produces_mm_values():
    # Degrees (angles) are calibration-independent by nature — the "never
    # fabricate mm" rule only applies to length/area quantities.
    part, circles = _build_part_and_circles()
    records = measurement.build_measurement_records(part, circles, profile=None)

    length_or_area_records = [r for r in records if r.unit_mm in ("mm", "mm²")]
    assert length_or_area_records, "expected at least one length/area record to check"
    for r in length_or_area_records:
        assert r.mm_value is None
        if r.px_value is not None:
            assert r.status == MM_UNAVAILABLE_STATUS


def test_calibration_conversion_is_exact_arithmetic():
    part, circles = _build_part_and_circles()
    img, _ = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200))
    profile = calibration.known_dimension_calibrate(img, known_mm=50.0, direction="horizontal")

    records = measurement.build_measurement_records(part, circles, profile=profile)
    width_record = next(r for r in records if r.feature == "Bounding Box Width")

    assert width_record.mm_value == pytest.approx(width_record.px_value * profile.mm_per_pixel, rel=1e-9)
    assert width_record.status == "Measured"


def test_area_uses_squared_scale():
    part, circles = _build_part_and_circles()
    img, _ = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200))
    profile = calibration.known_dimension_calibrate(img, known_mm=50.0, direction="horizontal")

    records = measurement.build_measurement_records(part, circles, profile=profile)
    area_record = next(r for r in records if r.feature == "Area")

    assert area_record.unit_mm == "mm²"
    assert area_record.mm_value == pytest.approx(area_record.px_value * profile.mm_per_pixel ** 2, rel=1e-9)


def test_tolerance_pass_fail_boundaries():
    from vision.types import MeasurementRecord
    records = [
        MeasurementRecord(feature="Width", category="Overall Dimensions", px_value=500.0,
                           mm_value=50.0, unit_px="px", unit_mm="mm", confidence="HIGH", status="Measured"),
    ]
    specs = {"Width": ToleranceSpec(feature="Width", nominal_mm=50.0, tolerance_mm=0.5)}
    results = measurement.apply_tolerances(records, specs)
    assert results[0].status == "PASS"

    specs_fail = {"Width": ToleranceSpec(feature="Width", nominal_mm=50.0, tolerance_mm=0.1)}
    records_fail = [
        MeasurementRecord(feature="Width", category="Overall Dimensions", px_value=500.0,
                           mm_value=50.2, unit_px="px", unit_mm="mm", confidence="HIGH", status="Measured"),
    ]
    results_fail = measurement.apply_tolerances(records_fail, specs_fail)
    assert results_fail[0].status == "FAIL"

    # Exactly at the boundary must PASS (inclusive range).
    records_boundary = [
        MeasurementRecord(feature="Width", category="Overall Dimensions", px_value=500.0,
                           mm_value=50.5, unit_px="px", unit_mm="mm", confidence="HIGH", status="Measured"),
    ]
    results_boundary = measurement.apply_tolerances(records_boundary, specs)
    assert results_boundary[0].status == "PASS"


def test_tolerance_missing_feature_is_not_applicable():
    specs = {"Nonexistent Feature": ToleranceSpec(feature="Nonexistent Feature", nominal_mm=10.0, tolerance_mm=1.0)}
    results = measurement.apply_tolerances([], specs)
    assert results[0].status == "N/A"
