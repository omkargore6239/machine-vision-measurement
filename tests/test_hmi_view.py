"""Unit tests for the HMI card's pure row-selection/lettering logic
(`ui.inspection_view.build_hmi_key_results`) -- no Streamlit, no image
pipeline, just: given already-computed results, does it pick the right
rows, in the right order, with the right (real, not fabricated) anchors?
"""
from ui.inspection_view import build_hmi_key_results, compute_hmi_verdict
from vision.shifter_fork_geometry import ShifterForkLandmarks
from vision.shifter_fork_measurement import ShifterForkAttributeResult
from vision.types import ParameterResult

BBOX = (10, 20, 100, 200)  # x, y, w, h -> fallback anchor (60, 120)


def _pr(parameter_id, name, status, value_kind="linear", measured_mm=10.0, debug=None):
    return ParameterResult(
        parameter_id=parameter_id, name=name, characteristic_type="x", value_kind=value_kind,
        nominal_mm=10.0, tolerance_display="±0.5 mm", lower_limit_mm=9.5, upper_limit_mm=10.5,
        measured_mm=measured_mm, unit="mm", deviation_mm=0.0, status=status, confidence="HIGH",
        method="test", debug=debug or {},
    )


def test_legacy_path_keeps_only_pass_fail_rows_in_order():
    results = [
        _pr("a", "A Param", "PASS"),
        _pr("b", "B Param", "NOT DETECTED"),
        _pr("c", "C Param", "FAIL"),
        _pr("d", "D Param", "NOT SPECIFIED"),
        _pr("e", "E Param", "SPECIFICATION CONFLICT"),
        _pr("f", "F Param", "INCOMPLETE"),
    ]
    rows = build_hmi_key_results(sf_results=None, sf_landmarks=None, inspection_results=results, part_bbox=BBOX)
    assert [r["label"] for r in rows] == ["A Param", "C Param"]
    assert [r["status"] for r in rows] == ["PASS", "FAIL"]
    assert [r["letter"] for r in rows] == ["A", "B"]


def test_legacy_path_caps_at_ten():
    results = [_pr(f"p{i}", f"Param {i}", "PASS") for i in range(14)]
    rows = build_hmi_key_results(sf_results=None, sf_landmarks=None, inspection_results=results, part_bbox=BBOX)
    assert len(rows) == 10
    assert rows[-1]["letter"] == "J"


def test_legacy_path_uses_debug_center_px_anchor():
    results = [_pr("a", "A", "PASS", debug={"center_px": (55.0, 77.0)})]
    rows = build_hmi_key_results(sf_results=None, sf_landmarks=None, inspection_results=results, part_bbox=BBOX)
    assert rows[0]["anchor"] == (55.0, 77.0)


def test_legacy_path_falls_back_to_bbox_center_when_debug_has_no_point():
    results = [_pr("a", "A", "PASS", debug={})]
    rows = build_hmi_key_results(sf_results=None, sf_landmarks=None, inspection_results=results, part_bbox=BBOX)
    assert rows[0]["anchor"] == (60.0, 120.0)  # bbox (10,20,100,200) center


def test_legacy_path_gives_width_and_height_distinct_edge_anchors():
    results = [
        _pr("w", "Overall Fork Width", "PASS", debug={"source_record": "Rotated Rect Width"}),
        _pr("h", "Overall Height", "PASS", debug={"source_record": "Rotated Rect Height"}),
    ]
    rows = build_hmi_key_results(sf_results=None, sf_landmarks=None, inspection_results=results, part_bbox=BBOX)
    assert rows[0]["anchor"] != rows[1]["anchor"]
    assert rows[0]["anchor"] != (60.0, 120.0)  # not the plain center either
    assert rows[1]["anchor"] != (60.0, 120.0)


def test_legacy_path_averages_two_point_debug_shapes():
    results = [_pr("a", "A", "PASS", debug={"tip_a_px": (0.0, 0.0), "tip_b_px": (10.0, 20.0)})]
    rows = build_hmi_key_results(sf_results=None, sf_landmarks=None, inspection_results=results, part_bbox=BBOX)
    assert rows[0]["anchor"] == (5.0, 10.0)


def test_legacy_path_value_string_uses_diameter_prefix():
    results = [_pr("a", "A", "PASS", value_kind="diameter", measured_mm=12.345)]
    rows = build_hmi_key_results(sf_results=None, sf_landmarks=None, inspection_results=results, part_bbox=BBOX)
    assert rows[0]["value_str"] == "Ø12.35 mm"


def _sf_row(feature, name, status, mapping="confirmed", in_verdict=True, measured_mm=5.0):
    return ShifterForkAttributeResult(
        feature=feature, name=name, drawing="x", mapping=mapping, in_verdict=in_verdict, unit="mm",
        nominal_mm=5.0, lower_mm=4.5, upper_mm=5.5, measured_mm=measured_mm, status=status,
    )


def test_shifter_fork_path_keeps_only_confirmed_in_verdict_rows():
    sf_results = [
        _sf_row("inner_opening", "Inner Fork Opening", "PASS", mapping="confirmed", in_verdict=True),
        _sf_row("axis_to_bore_x", "Bore Offset", "FAIL", mapping="assumed", in_verdict=True),
        _sf_row("bore_diameter", "Bore Diameter", "FAIL", mapping="confirmed", in_verdict=False),
        _sf_row("axis_to_left_inner", "Fork Half Opening", "PASS", mapping="confirmed", in_verdict=True),
    ]
    rows = build_hmi_key_results(sf_results=sf_results, sf_landmarks=None, inspection_results=[], part_bbox=BBOX)
    assert [r["label"] for r in rows] == ["Inner Fork Opening", "Fork Half Opening"]
    assert [r["letter"] for r in rows] == ["A", "B"]


def test_shifter_fork_path_uses_landmark_anchor_for_known_feature():
    lm = ShifterForkLandmarks(
        left_inner_line={"point": (0.0, 0.0), "direction": (0.0, 1.0)},
        right_inner_line={"point": (20.0, 0.0), "direction": (0.0, 1.0)},
    )
    sf_results = [_sf_row("inner_opening", "Inner Fork Opening", "PASS")]
    rows = build_hmi_key_results(sf_results=sf_results, sf_landmarks=lm, inspection_results=[], part_bbox=BBOX)
    assert rows[0]["anchor"] == (10.0, 0.0)  # midpoint of the two inner-line points


def test_shifter_fork_path_falls_back_to_bbox_center_without_landmarks():
    sf_results = [_sf_row("inner_opening", "Inner Fork Opening", "PASS")]
    rows = build_hmi_key_results(sf_results=sf_results, sf_landmarks=None, inspection_results=[], part_bbox=BBOX)
    assert rows[0]["anchor"] == (60.0, 120.0)


# --- compute_hmi_verdict -------------------------------------------------------------
# Judged from exactly the rows the card shows, not a separate full-spec
# aggregate -- so the badge never disagrees with what's visibly on screen.

def test_verdict_is_pass_when_all_shown_rows_pass():
    key_results = [{"status": "PASS"}, {"status": "PASS"}]
    assert compute_hmi_verdict(key_results) == ("pass", "PASS")


def test_verdict_is_fail_when_any_shown_row_fails():
    key_results = [{"status": "PASS"}, {"status": "FAIL"}]
    key, label = compute_hmi_verdict(key_results)
    assert key == "fail"
    assert "FAIL" in label


def test_verdict_is_incomplete_when_nothing_to_show():
    key, label = compute_hmi_verdict([])
    assert key == "incomplete"
    assert "INCOMPLETE" in label
