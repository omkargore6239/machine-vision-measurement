import pytest

from tests.fixtures import (
    make_axis_aligned_part, make_busy_product_photo, make_circular_flange_part, make_fork_bracket_part,
)
from vision import calibration, geometry, measurement, segmentation
from vision.inspection_spec import INSPECTION_SPEC
from vision.preprocessing import to_gray
from vision.types import (
    MM_UNAVAILABLE_STATUS, ToleranceDefinition, ToleranceSpec, bilateral, negative_only, not_specified, positive_only,
    PARAM_STATUS_FAIL, PARAM_STATUS_INCOMPLETE, PARAM_STATUS_NOT_DETECTED, PARAM_STATUS_NOT_SPECIFIED,
    PARAM_STATUS_PASS, PARAM_STATUS_SPEC_CONFLICT,
)


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


def _flange_part_and_circles():
    img, gt = make_circular_flange_part()
    part = segmentation.build_detected_part(img)
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    return part, circles, gt


def test_classify_and_label_circles_finds_the_bore():
    part, circles, gt = _flange_part_and_circles()
    circles = measurement.classify_and_label_circles(circles, part)

    bores = [c for c in circles if c.feature_type == "bore"]
    holes = [c for c in circles if c.feature_type == "hole"]
    assert len(bores) == 1
    assert bores[0].label == "Center Bore" and bores[0].short_id == "CB"
    assert len(holes) == 6
    assert sorted(c.label for c in holes) == [f"Hole {i}" for i in range(1, 7)]


def test_classify_and_label_circles_single_hole_is_not_a_bore():
    # A single small hole with nothing to compare against shouldn't be
    # mislabeled as "the bore" just because it's the only (and therefore
    # "largest") feature — the size/position heuristic still has to hold.
    holes = [(320, 260, 15)]
    img, gt = make_axis_aligned_part(rect_xywh=(150, 150, 500, 400), holes=holes)
    part = segmentation.build_detected_part(img)
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    circles = measurement.classify_and_label_circles(circles, part)

    assert len(circles) == 1
    assert circles[0].feature_type == "hole"
    assert circles[0].label == "Hole 1"


def test_build_feature_summary_table_columns_and_uncalibrated_units():
    part, circles, gt = _flange_part_and_circles()
    circles = measurement.classify_and_label_circles(circles, part)
    x, y, w, h = part.bbox
    origin = (float(x), float(y))

    rows = measurement.build_feature_summary_table(circles, profile=None, origin_px=origin)
    assert len(rows) == len(circles)
    for row in rows:
        assert set(row.keys()) == {"Feature", "Type", "X", "Y", "Diameter", "Unit", "Confidence", "Tolerance", "Result"}
        assert row["Unit"] == "px"
        assert row["Tolerance"] == "—"
        assert row["Result"] == "N/A"
    assert any(row["Type"] == "Central Bore" for row in rows)
    assert sum(row["Type"] == "Hole" for row in rows) == 6


def test_build_feature_summary_table_calibrated_units_and_tolerance():
    part, circles, gt = _flange_part_and_circles()
    circles = measurement.classify_and_label_circles(circles, part)
    x, y, w, h = part.bbox
    origin = (float(x), float(y))

    img, _ = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200))
    profile = calibration.known_dimension_calibrate(img, known_mm=50.0, direction="horizontal")

    records = measurement.build_measurement_records(part, circles, profile, origin)
    bore_label = next(c.label for c in circles if c.feature_type == "bore")
    tol_specs = {f"{bore_label} Equivalent Diameter": ToleranceSpec(feature=f"{bore_label} Equivalent Diameter", nominal_mm=1000.0, tolerance_mm=0.5)}
    tol_results = measurement.apply_tolerances(records, tol_specs)

    rows = measurement.build_feature_summary_table(circles, profile, origin, tol_results)
    for row in rows:
        assert row["Unit"] == "mm"
        assert isinstance(row["Diameter"], float)

    bore_row = next(r for r in rows if r["Type"] == "Central Bore")
    assert bore_row["Tolerance"] == "±0.5"
    assert bore_row["Result"] == "FAIL"  # nominal of 1000mm is nowhere near the real bore


# --- product-specific inspection spec (fork bracket) --------------------------------

def _fork_part_circles_records(profile=None):
    img, gt = make_fork_bracket_part()
    part = segmentation.build_detected_part(img)
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    circles = measurement.classify_and_label_circles(circles, part)
    x, y, w, h = part.bbox
    records = measurement.build_measurement_records(part, circles, profile, (float(x), float(y)))
    return part, circles, records, gt


def _fork_calibration_profile():
    img, _ = make_fork_bracket_part()
    return calibration.known_dimension_calibrate(img, known_mm=50.0, direction="horizontal")


def test_inspection_spec_covers_all_configured_parameters():
    # Active spec is the original 10 only -- the 12 additional-sheet
    # parameters (ADDITIONAL_SHEET_SPEC) are deliberately excluded from
    # evaluation per the operator's 2026-09-30 "measure only these 10"
    # request, though kept in vision/inspection_spec.py for later reuse.
    part, circles, records, gt = _fork_part_circles_records(profile=None)
    results = measurement.evaluate_inspection_spec(part, circles, records, profile=None)
    assert len(results) == len(INSPECTION_SPEC) == 10
    assert {r.parameter_id for r in results} == {s.parameter_id for s in INSPECTION_SPEC}


def test_inspection_spec_uncalibrated_is_incomplete_not_fail():
    # A real pixel measurement exists for every geometrically-detectable
    # parameter on this fixture, but with no calibration active none of
    # them can become an mm PASS/FAIL — must be INCOMPLETE, never a guessed
    # PASS or a silently-fabricated FAIL. The parameters with a documented
    # `not_detectable_reason` are NOT DETECTED regardless of calibration,
    # and Overall Fork Width is always SPECIFICATION CONFLICT (checked
    # before calibration even matters) unless explicitly overridden.
    part, circles, records, gt = _fork_part_circles_records(profile=None)
    results = measurement.evaluate_inspection_spec(part, circles, records, profile=None)
    by_id = {r.parameter_id: r for r in results}

    not_detectable_ids = {s.parameter_id for s in INSPECTION_SPEC if s.not_detectable_reason is not None}
    for pid in not_detectable_ids:
        assert by_id[pid].status == PARAM_STATUS_NOT_DETECTED, pid

    assert by_id["overall_fork_width"].status == "SPECIFICATION CONFLICT"

    other_ids = {s.parameter_id for s in INSPECTION_SPEC} - not_detectable_ids - {"part_thickness", "overall_fork_width"}
    for pid in other_ids:
        assert by_id[pid].status == PARAM_STATUS_INCOMPLETE, f"{pid} -> {by_id[pid].status}"
    assert all(r.measured_mm is None for r in results)


def test_inspection_spec_part_thickness_always_incomplete():
    part, circles, records, gt = _fork_part_circles_records(profile=None)
    results = measurement.evaluate_inspection_spec(part, circles, records, profile=None)
    thickness = next(r for r in results if r.parameter_id == "part_thickness")
    assert thickness.status == PARAM_STATUS_INCOMPLETE
    assert thickness.measured_mm is None
    assert thickness.reason  # a real explanation, not a blank

    # Must ALSO be INCOMPLETE even when calibrated -- image content never
    # changes this; only a genuinely different capture setup could.
    profile = _fork_calibration_profile()
    part2, circles2, records2, _ = _fork_part_circles_records(profile=profile)
    results2 = measurement.evaluate_inspection_spec(part2, circles2, records2, profile=profile)
    thickness2 = next(r for r in results2 if r.parameter_id == "part_thickness")
    assert thickness2.status == PARAM_STATUS_INCOMPLETE
    assert thickness2.measured_mm is None


def test_inspection_spec_tolerance_limit_math():
    profile = _fork_calibration_profile()
    part, circles, records, gt = _fork_part_circles_records(profile=profile)

    overrides = {"main_hole_diameter": (10.0, 2.0)}
    results = measurement.evaluate_inspection_spec(part, circles, records, profile, overrides)
    main = next(r for r in results if r.parameter_id == "main_hole_diameter")
    assert main.nominal_mm == 10.0
    assert main.tolerance_display == "±2 mm"
    assert main.lower_limit_mm == pytest.approx(8.0)
    assert main.upper_limit_mm == pytest.approx(12.0)


def test_inspection_spec_pass_when_measured_matches_nominal():
    profile = _fork_calibration_profile()
    part, circles, records, gt = _fork_part_circles_records(profile=profile)
    baseline = measurement.evaluate_inspection_spec(part, circles, records, profile)
    main = next(r for r in baseline if r.parameter_id == "main_hole_diameter")
    assert main.measured_mm is not None

    # boundary case: nominal set exactly to the measured value -> PASS
    overrides = {"main_hole_diameter": (main.measured_mm, 0.05)}
    results = measurement.evaluate_inspection_spec(part, circles, records, profile, overrides)
    main2 = next(r for r in results if r.parameter_id == "main_hole_diameter")
    assert main2.status == PARAM_STATUS_PASS
    assert main2.deviation_mm == pytest.approx(0.0, abs=1e-6)


def test_inspection_spec_fail_when_measured_outside_tolerance():
    profile = _fork_calibration_profile()
    part, circles, records, gt = _fork_part_circles_records(profile=profile)
    overrides = {"main_hole_diameter": (1.0, 0.1)}  # deliberately unreachable nominal
    results = measurement.evaluate_inspection_spec(part, circles, records, profile, overrides)
    main = next(r for r in results if r.parameter_id == "main_hole_diameter")
    assert main.status == PARAM_STATUS_FAIL
    assert main.measured_mm is not None
    assert main.deviation_mm is not None and main.deviation_mm > 0


# --- tolerance engine: bilateral / positive-only / negative-only / none -------------

def test_tolerance_bilateral_limits_and_display():
    t = bilateral(0.30)
    assert t.limits(50.0) == pytest.approx((49.70, 50.30))
    assert t.display() == "±0.3 mm"
    assert t.has_tolerance


def test_tolerance_positive_only_limits_and_display():
    t = positive_only(0.30)
    lower, upper = t.limits(94.20)
    assert lower == pytest.approx(94.20)
    assert upper == pytest.approx(94.50)
    assert t.display() == "+0.3/-0 mm"


def test_tolerance_negative_only_limits_and_display():
    # Not currently used by any spec entry, but must work -- the brief
    # explicitly asks to support it "even if not currently used".
    t = negative_only(0.30)
    lower, upper = t.limits(50.0)
    assert lower == pytest.approx(49.70)
    assert upper == pytest.approx(50.0)
    assert t.display() == "+0/-0.3 mm"


def test_tolerance_none_has_no_limits_and_is_not_specified():
    t = not_specified()
    assert t.limits(62.00) == (None, None)
    assert t.display() == "NOT SPECIFIED"
    assert not t.has_tolerance


def test_tolerance_kind_none_is_distinct_from_zero_width():
    # A "none" tolerance is a different state from "zero-width" (must match
    # exactly) -- .limits() must return (None, None), not (nominal, nominal).
    t = ToleranceDefinition("none", plus_mm=0.0, minus_mm=0.0)
    assert t.limits(10.0) == (None, None)


# --- additional-sheet spec (inactive, kept for reuse): structural checks only ------
# ADDITIONAL_SHEET_SPEC is deliberately excluded from INSPECTION_SPEC/live
# evaluation (operator's 2026-09-30 "measure only these 10" request), so
# these check the ParameterSpec data itself, not `evaluate_inspection_spec`
# output -- there isn't any for these parameters anymore (see
# test_additional_sheet_parameters_are_excluded_from_active_evaluation).

def test_additional_sheet_radius_parameters_are_defined_as_radius_not_diameter():
    from vision.inspection_spec import ADDITIONAL_SHEET_SPEC
    outer = next(s for s in ADDITIONAL_SHEET_SPEC if s.parameter_id == "outer_arc_radius")
    assert outer.value_kind == "radius"


def test_additional_sheet_outer_arc_radius_and_arc_radius_have_no_tolerance():
    from vision.inspection_spec import ADDITIONAL_SHEET_SPEC
    by_id = {s.parameter_id: s for s in ADDITIONAL_SHEET_SPEC}
    for pid in ("outer_arc_radius", "arc_radius"):
        assert by_id[pid].tolerance.display() == "NOT SPECIFIED"
        assert not by_id[pid].tolerance.has_tolerance


def test_inspection_spec_overall_fork_width_is_specification_conflict_by_default():
    profile = _fork_calibration_profile()
    part, circles, records, gt = _fork_part_circles_records(profile=profile)
    results = measurement.evaluate_inspection_spec(part, circles, records, profile)
    width = next(r for r in results if r.parameter_id == "overall_fork_width")
    assert width.status == PARAM_STATUS_SPEC_CONFLICT
    # the measured value is still shown -- conflict withholds judgment, not data
    assert width.measured_mm is not None
    # both candidate nominal/tolerance values must be visible, not silently dropped
    assert "105" in width.conflict_note and "104" in width.conflict_note
    assert "confirmation" in width.conflict_note.lower()


def test_inspection_spec_overall_fork_width_conflict_resolved_by_explicit_override():
    # An operator override IS the "engineering confirmation" the conflict
    # is waiting for -- it must resolve to a normal PASS/FAIL, not conflict.
    profile = _fork_calibration_profile()
    part, circles, records, gt = _fork_part_circles_records(profile=profile)
    baseline = measurement.evaluate_inspection_spec(part, circles, records, profile)
    width0 = next(r for r in baseline if r.parameter_id == "overall_fork_width")
    assert width0.measured_mm is not None

    overrides = {"overall_fork_width": (width0.measured_mm, 0.5)}
    results = measurement.evaluate_inspection_spec(part, circles, records, profile, overrides)
    width = next(r for r in results if r.parameter_id == "overall_fork_width")
    assert width.status == PARAM_STATUS_PASS
    assert width.conflict_note == ""


def test_additional_sheet_ambiguous_parameters_have_specific_not_detectable_reasons():
    # Structural check on the (inactive) ADDITIONAL_SHEET_SPEC data itself --
    # each of these was already not_detectable_reason-gated even back when
    # active, so excluding the whole sheet from evaluation doesn't change
    # what they'd report; this just confirms the reasons are still real,
    # specific text, never blank, in case the sheet is reactivated later.
    from vision.inspection_spec import ADDITIONAL_SHEET_SPEC
    by_id = {s.parameter_id: s for s in ADDITIONAL_SHEET_SPEC}

    ambiguous = [
        "inner_fork_opening", "fork_half_width", "reference_distance", "overall_arc_height",
        "related_arc_height", "inner_arc_radius", "transition_radius",
        "boss_upper_geometry_a", "boss_upper_geometry_b", "boss_upper_geometry_c",
    ]
    for pid in ambiguous:
        assert by_id[pid].not_detectable_reason, pid  # a specific, real explanation -- never blank


def test_additional_sheet_parameters_are_excluded_from_active_evaluation():
    # The operator's 2026-09-30 "measure only these 10" request: none of
    # ADDITIONAL_SHEET_SPEC's parameter_ids should appear in a live
    # evaluate_inspection_spec() run at all.
    from vision.inspection_spec import ADDITIONAL_SHEET_SPEC
    part, circles, records, gt = _fork_part_circles_records(profile=None)
    results = measurement.evaluate_inspection_spec(part, circles, records, profile=None)
    active_ids = {r.parameter_id for r in results}
    additional_ids = {s.parameter_id for s in ADDITIONAL_SHEET_SPEC}
    assert active_ids.isdisjoint(additional_ids)


def test_inspection_spec_existing_ten_parameters_still_present_and_unchanged_nominals():
    # This round must not delete or renumber the original 10. Their
    # nominal/tolerance values are the operator-confirmed 2026-09-30
    # "Tentative Nominal" table (a real reference, not a placeholder) --
    # a same-day earlier attempt to sample-calibrate 4 of them from a real
    # photo was explicitly reverted once the operator provided this table.
    part, circles, records, gt = _fork_part_circles_records(profile=None)
    results = measurement.evaluate_inspection_spec(part, circles, records, profile=None)
    by_id = {r.parameter_id: r for r in results}
    expected_nominals = {
        "inner_arc_diameter": 92.00, "main_hole_diameter": 20.00, "top_boss_hole_diameter": 16.00,
        "overall_fork_width": 105.00, "fork_arm_length": 48.00, "fork_tip_gap": 72.00,
        "overall_height": 82.00, "part_thickness": 12.00, "hole_center_distance": 78.00,
        "fork_tip_thickness": 8.00,
    }
    for pid, nominal in expected_nominals.items():
        assert by_id[pid].nominal_mm == pytest.approx(nominal), pid


def test_inspection_spec_not_detected_when_feature_genuinely_absent():
    # A plain rectangle with a single hole has no second hole and no fork
    # opening at all -- these parameters must report NOT DETECTED (the
    # underlying feature isn't there), never a fabricated value.
    holes = [(320, 260, 20)]
    img, gt = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=holes)
    part = segmentation.build_detected_part(img)
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    circles = measurement.classify_and_label_circles(circles, part)
    x, y, w, h = part.bbox
    records = measurement.build_measurement_records(part, circles, None, (float(x), float(y)))

    results = measurement.evaluate_inspection_spec(part, circles, records, None)
    by_id = {r.parameter_id: r for r in results}
    assert by_id["top_boss_hole_diameter"].status == PARAM_STATUS_NOT_DETECTED
    assert by_id["fork_tip_gap"].status == PARAM_STATUS_NOT_DETECTED
    assert by_id["inner_arc_diameter"].status == PARAM_STATUS_NOT_DETECTED
    assert by_id["fork_arm_length"].status == PARAM_STATUS_NOT_DETECTED
    assert by_id["fork_tip_thickness"].status == PARAM_STATUS_NOT_DETECTED
    assert by_id["hole_center_distance"].status == PARAM_STATUS_NOT_DETECTED
    # the one hole that IS present should still be found (just uncalibrated)
    assert by_id["main_hole_diameter"].status == PARAM_STATUS_INCOMPLETE


def test_inspection_spec_hole_center_distance_uses_actual_centers_not_diameter():
    profile = _fork_calibration_profile()
    part, circles, records, gt = _fork_part_circles_records(profile=profile)
    results = measurement.evaluate_inspection_spec(part, circles, records, profile)
    by_id = {r.parameter_id: r for r in results}
    dist = by_id["hole_center_distance"]
    assert dist.measured_mm is not None

    main_hole = next((c for c in circles if c.feature_type == "bore"), max(circles, key=lambda c: c.r))
    boss = next(c for c in circles if c is not main_hole)
    expected_mm = geometry.euclidean_distance((main_hole.cx, main_hole.cy), (boss.cx, boss.cy)) * profile.mm_per_pixel
    assert dist.measured_mm == pytest.approx(expected_mm, rel=1e-6)
    # sanity: must not equal a diameter (a formula bug using the wrong
    # quantity) or an edge-to-edge distance (diameter difference off by a
    # large, obviously-wrong margin)
    assert dist.measured_mm != pytest.approx(by_id["main_hole_diameter"].measured_mm, rel=1e-3)


def test_inspection_spec_decorative_marks_not_used_as_hole_parameters():
    # Reuses the existing adversarial fixture (printed text + vent grille +
    # barcode block, on top of real holes) -- decorative marks must not
    # leak into Main Hole / Top Boss Hole.
    real_holes = [(500, 450, 20), (700, 450, 14)]
    img, gt = make_busy_product_photo(real_holes=real_holes)
    part = segmentation.build_detected_part(img)
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    assert len(circles) == len(real_holes)  # confirms no decorative mark passed detect_holes itself

    circles = measurement.classify_and_label_circles(circles, part)
    x, y, w, h = part.bbox
    records = measurement.build_measurement_records(part, circles, None, (float(x), float(y)))
    results = measurement.evaluate_inspection_spec(part, circles, records, None)
    by_id = {r.parameter_id: r for r in results}
    assert by_id["main_hole_diameter"].status != PARAM_STATUS_NOT_DETECTED
    assert by_id["top_boss_hole_diameter"].status != PARAM_STATUS_NOT_DETECTED
