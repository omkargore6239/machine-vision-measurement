"""Data sanity for the 3 real shifter-fork recipes -- catches typos in the
ported spec sheet before they ever reach the measurement pipeline."""
from vision.shifter_fork_measurement import KNOWN_FEATURES
from vision.shifter_fork_recipes import SHIFTER_FORK_RECIPES


def test_three_recipes_present():
    assert set(SHIFTER_FORK_RECIPES) == {
        "OP-T02-013-A-00-010", "OP-T02-023-A-00-010", "OP-T02-015-A-00-010",
    }


def test_every_attribute_nominal_within_its_own_limits():
    for recipe in SHIFTER_FORK_RECIPES.values():
        for attr in recipe.attributes:
            assert attr.min_mm <= attr.nominal_mm <= attr.max_mm, (
                f"{recipe.part_number}/{attr.name}: nominal {attr.nominal_mm} "
                f"outside [{attr.min_mm}, {attr.max_mm}]"
            )


def test_every_feature_key_is_wired_to_a_landmark():
    for recipe in SHIFTER_FORK_RECIPES.values():
        for attr in recipe.attributes:
            assert attr.feature in KNOWN_FEATURES, (
                f"{recipe.part_number}/{attr.name}: unknown feature {attr.feature!r}"
            )


def test_bore_diameter_is_confirmed_but_excluded_from_verdict():
    for recipe in SHIFTER_FORK_RECIPES.values():
        bore = next(a for a in recipe.attributes if a.feature == "bore_diameter")
        assert bore.mapping == "confirmed"
        assert bore.in_verdict is False


def test_mapping_values_are_only_confirmed_or_assumed():
    for recipe in SHIFTER_FORK_RECIPES.values():
        for attr in recipe.attributes:
            assert attr.mapping in ("confirmed", "assumed")


def test_side_view_rows_are_present_and_never_decide_the_verdict():
    for recipe in SHIFTER_FORK_RECIPES.values():
        side = [a for a in recipe.attributes if a.needs_view]
        assert {a.feature for a in side} == {"cross_hole_5h10", "pad_thickness", "tip_thickness_side"}
        assert all(a.in_verdict is False for a in side)
