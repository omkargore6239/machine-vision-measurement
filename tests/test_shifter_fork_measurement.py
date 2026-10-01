"""Per-recipe evaluation tests against all 3 real shifter-fork recipes: a
confirmed+in_verdict row failing (or being unmeasurable) must reject the
part; an assumed row or an in_verdict=False row (the bore) must NEVER
affect that boolean even when it would itself read as a FAIL; and a hard
`reject_reason` from geometry (no bore, no part, etc.) must force
reject=True regardless of any individual row.

Two kinds of tests here, deliberately separated:

  - Verdict-LOGIC tests (`_exact_landmarks_for_recipe`) construct a
    `ShifterForkLandmarks` directly with exact, hand-computed geometry
    (bypassing image segmentation entirely) -- these prove
    `evaluate_shifter_fork`'s dispatch/mapping/in_verdict/reject rules are
    correct in isolation, at effectively infinite measurement precision.
  - End-to-end pipeline tests (`_build_for_recipe`) go through the real
    image -> segmentation -> geometry pipeline on the synthetic fixture,
    the same way a real photo would. These proved the wiring is correct
    and values are plausible, but are NOT used to assert PASS against the
    tightest real tolerances (some as tight as +-0.15mm): the shared
    `vision.segmentation` candidate-mask scoring can select a mildly
    dilated boundary over a pixel-precise one for some part proportions
    (confirmed directly during this work -- a `canny_morphology` mask
    candidate outscoring the plain-threshold ones by a few percent of the
    part's size), which is a pre-existing, out-of-scope characteristic of
    the shared segmentation module, not a bug in this recipe's own
    geometry/measurement code. Achieving the real tolerance bands end to
    end needs validation (and likely tuning) against real camera photos,
    not just this synthetic fixture -- flagged here and in the task report
    rather than silently asserted away.
"""
import pytest

from tests.fixtures import make_shifter_fork_part
from vision import geometry, segmentation
from vision.preprocessing import to_gray
from vision.shifter_fork_geometry import ShifterForkLandmarks, extract_shifter_fork_landmarks
from vision.shifter_fork_measurement import evaluate_shifter_fork
from vision.shifter_fork_recipes import SHIFTER_FORK_RECIPES
from vision.types import CalibrationProfile, CircleFeature

# High enough that sub-pixel fit noise (a fraction of a px) stays well
# under the tightest real tolerance band (+-0.15mm on some attributes) --
# a real inspection system would need comparable effective resolution at
# the part for those tolerances to be judged reliably at all.
SCALE_PX_PER_MM = 8.0


def _profile(scale=SCALE_PX_PER_MM):
    return CalibrationProfile(
        profile_id="test", name="test", method="known_dimension",
        created_at="2026-01-01T00:00:00", confidence="HIGH", mm_per_pixel=1.0 / scale,
    )


def _build_for_recipe(recipe_id: str, scale=SCALE_PX_PER_MM, **fixture_overrides):
    recipe = SHIFTER_FORK_RECIPES[recipe_id]
    by_feature = {a.feature: a for a in recipe.attributes}

    inner_opening_mm = by_feature["inner_opening"].nominal_mm
    inner_opening_px = inner_opening_mm * scale

    if "outer_width" in by_feature:
        outer_width_mm = by_feature["outer_width"].nominal_mm
        prong_wall_px = (outer_width_mm - inner_opening_mm) / 2.0 * scale
    else:
        prong_wall_px = 15.0 * scale

    # Every other fixture dimension must scale consistently too (mm-based
    # defaults * scale), or a fixed default in raw px at a higher `scale`
    # would badly distort the proportions (e.g. too little boss material
    # for the bore to fit).
    kwargs = dict(
        inner_opening_px=inner_opening_px,
        prong_wall_px=prong_wall_px,
        straight_side_len_px=20.0 * scale,
        boss_straight_height_px=15.0 * scale,
        bore_radius_px=7.0 * scale,
        bore_transverse_offset_px=-10.0 * scale,
    )
    kwargs.update(fixture_overrides)

    img, gt = make_shifter_fork_part(**kwargs)
    part = segmentation.build_detected_part(img)
    assert part is not None
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour)
    landmarks = extract_shifter_fork_landmarks(part, circles)
    return landmarks, recipe, gt


def _exact_landmarks_for_recipe(
    recipe, scale: float = 10.0, bore_diameter_mm: float | None = None, bore_offset_mm: float = 0.0,
) -> ShifterForkLandmarks:
    """Builds landmarks directly from a recipe's own nominal values (no
    image/segmentation involved), for testing `evaluate_shifter_fork`'s
    verdict logic at exact, controlled precision. `axis_direction` here is
    (0,-1); the perpendicular is (1,0), so a positive `bore_offset_mm`
    moves the bore in +x."""
    by_feature = {a.feature: a for a in recipe.attributes}
    half_opening_mm = by_feature["inner_opening"].nominal_mm / 2.0
    inner_arc_radius_mm = (
        by_feature["inner_arc_radius"].nominal_mm if "inner_arc_radius" in by_feature else half_opening_mm
    )
    outer_width_mm = (
        by_feature["outer_width"].nominal_mm if "outer_width" in by_feature else half_opening_mm * 2.0 + 20.0
    )
    if bore_diameter_mm is None:
        bore_diameter_mm = by_feature["bore_diameter"].nominal_mm

    axis_origin = (0.0, 0.0)
    axis_direction = (0.0, -1.0)

    half_opening_px = half_opening_mm * scale
    left_inner_line = {"point": (-half_opening_px, 0.0), "direction": (0.0, 1.0)}
    right_inner_line = {"point": (half_opening_px, 0.0), "direction": (0.0, 1.0)}

    inner_arc = {"cx": 0.0, "cy": 0.0, "radius_px": inner_arc_radius_mm * scale,
                 "inlier_fraction": 1.0, "angular_span_deg": 180.0, "rms_px": 0.0, "n_run_points": 100}
    outer_arc = {"cx": 0.0, "cy": -200.0 * scale, "radius_px": 60.0 * scale,
                 "inlier_fraction": 1.0, "angular_span_deg": 180.0, "rms_px": 0.0, "n_run_points": 100}

    tip_point = (inner_arc["cx"] + inner_arc["radius_px"] * axis_direction[0],
                 inner_arc["cy"] + inner_arc["radius_px"] * axis_direction[1])
    top_point = (outer_arc["cx"] + outer_arc["radius_px"] * axis_direction[0],
                 outer_arc["cy"] + outer_arc["radius_px"] * axis_direction[1])

    outer_half_width_px = outer_width_mm / 2.0 * scale
    bore = CircleFeature(
        circle_id=1, cx=bore_offset_mm * scale, cy=tip_point[1] * 0.5, r=bore_diameter_mm / 2.0 * scale,
        method="test", circularity=1.0, confidence="HIGH", feature_type="bore",
    )

    return ShifterForkLandmarks(
        reject_reason=None, axis_origin=axis_origin, axis_direction=axis_direction, axis_angle_deg=-90.0,
        tip_point=tip_point, top_point=top_point,
        left_inner_line=left_inner_line, right_inner_line=right_inner_line,
        inner_arc=inner_arc, outer_arc_left=outer_arc, outer_arc_right=outer_arc,
        bore=bore, outer_width_px=outer_half_width_px * 2.0,
        outer_left_transverse_px=-outer_half_width_px, outer_right_transverse_px=outer_half_width_px,
    )


@pytest.mark.parametrize("recipe_id", list(SHIFTER_FORK_RECIPES))
def test_confirmed_attributes_pass_on_an_exactly_nominal_part(recipe_id):
    recipe = SHIFTER_FORK_RECIPES[recipe_id]
    landmarks = _exact_landmarks_for_recipe(recipe)
    results, reject, reason = evaluate_shifter_fork(landmarks, recipe, _profile(scale=10.0))

    confirmed = [r for r in results if r.mapping == "confirmed" and r.in_verdict]
    assert confirmed, "every recipe must have at least one verdict-deciding row"
    for r in confirmed:
        assert r.status == "PASS", f"{recipe_id}/{r.name}: expected PASS, got {r.status} ({r.reason})"
    assert reject is False, reason


@pytest.mark.parametrize("recipe_id", list(SHIFTER_FORK_RECIPES))
def test_every_attribute_is_evaluated_and_reported(recipe_id):
    landmarks, recipe, gt = _build_for_recipe(recipe_id)
    results, _, _ = evaluate_shifter_fork(landmarks, recipe, _profile())
    assert len(results) == len(recipe.attributes)
    for r in results:
        assert r.status in ("PASS", "FAIL", "INCOMPLETE", "NOT DETECTED")


@pytest.mark.parametrize("recipe_id", list(SHIFTER_FORK_RECIPES))
def test_end_to_end_image_pipeline_produces_plausible_values(recipe_id):
    # Not a tight-tolerance check (see module docstring) -- proves the
    # image -> segmentation -> geometry -> recipe-evaluation wiring is
    # correct and produces sane, roughly-nominal numbers end to end.
    landmarks, recipe, gt = _build_for_recipe(recipe_id)
    assert landmarks.reject_reason is None
    results, _, _ = evaluate_shifter_fork(landmarks, recipe, _profile())
    for r in results:
        if r.mapping == "confirmed" and r.measured_mm is not None:
            assert abs(r.measured_mm - r.nominal_mm) / r.nominal_mm < 0.10, (
                f"{recipe_id}/{r.name}: measured {r.measured_mm} wildly far from nominal {r.nominal_mm}"
            )


@pytest.mark.parametrize("recipe_id", list(SHIFTER_FORK_RECIPES))
def test_bore_diameter_never_affects_verdict_even_when_wrong(recipe_id):
    # bore_diameter is "confirmed" but in_verdict=False in every recipe --
    # a wildly wrong bore must still be reported, but never reject the part.
    recipe = SHIFTER_FORK_RECIPES[recipe_id]
    landmarks = _exact_landmarks_for_recipe(recipe, bore_diameter_mm=3.0)  # way off from ~14mm nominal
    results, reject, reason = evaluate_shifter_fork(landmarks, recipe, _profile(scale=10.0))

    bore_row = next(r for r in results if r.feature == "bore_diameter")
    assert bore_row.mapping == "confirmed" and bore_row.in_verdict is False
    assert bore_row.status == "FAIL"  # honestly reported...
    assert reject is False, "...but must never affect the accept/reject verdict"


@pytest.mark.parametrize("recipe_id", list(SHIFTER_FORK_RECIPES))
def test_assumed_rows_never_affect_verdict_even_when_wrong(recipe_id):
    # Push the bore far off-axis so every *_bore_x / bore_angle_deg
    # "assumed" attribute reads as a clear FAIL, while every "confirmed"
    # geometry (inner opening, half-opening, outer width) stays exactly
    # nominal.
    recipe = SHIFTER_FORK_RECIPES[recipe_id]
    landmarks = _exact_landmarks_for_recipe(recipe, bore_offset_mm=-30.0)
    results, reject, reason = evaluate_shifter_fork(landmarks, recipe, _profile(scale=10.0))

    assumed_rows = [r for r in results if r.mapping == "assumed"]
    assert assumed_rows
    confirmed = [r for r in results if r.mapping == "confirmed" and r.in_verdict]
    assert all(r.status == "PASS" for r in confirmed)
    assert reject is False


def test_reject_true_when_a_confirmed_attribute_fails():
    # Build recipe 1's part but with a badly wrong inner opening (a real
    # out-of-tolerance part) -- inner_opening is "confirmed" and must drive
    # reject=True.
    landmarks, recipe, gt = _build_for_recipe("OP-T02-013-A-00-010", inner_opening_px=40.0 * SCALE_PX_PER_MM)
    results, reject, reason = evaluate_shifter_fork(landmarks, recipe, _profile())

    opening_row = next(r for r in results if r.feature == "inner_opening")
    assert opening_row.status == "FAIL"
    assert reject is True
    assert "Inner Fork Opening" in reason or "FAIL" in reason


def test_reject_true_when_landmarks_have_a_hard_failure():
    landmarks, recipe, gt = _build_for_recipe("OP-T02-013-A-00-010", bore_radius_px=0.0)
    assert landmarks.reject_reason is not None  # no bore found
    results, reject, reason = evaluate_shifter_fork(landmarks, recipe, _profile())

    assert reject is True
    assert reason == landmarks.reject_reason
    # every row reports the same hard-failure reason, not a fabricated value
    for r in results:
        assert r.measured_mm is None
        assert r.status == "NOT DETECTED"


def test_reject_true_when_no_calibration_active():
    # A real pixel measurement exists but can't be converted to mm -- must
    # never guess a PASS.
    landmarks, recipe, gt = _build_for_recipe("OP-T02-013-A-00-010")
    results, reject, reason = evaluate_shifter_fork(landmarks, recipe, profile=None)

    confirmed = [r for r in results if r.mapping == "confirmed" and r.in_verdict]
    assert all(r.status == "INCOMPLETE" for r in confirmed)
    assert reject is True
