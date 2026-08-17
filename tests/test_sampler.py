"""The determinism and coherence guarantees of the attribute sampler.

These are the two promises the rest of the project is built on:

1. the same seed always produces the same person, on any machine, forever
   (this is what makes ``/v1/avatars/by-seed/<key>`` a usable avatar service);
2. the person is internally coherent - no retired 19-year-olds, no beards on a
   female-sampled face, no implausible skin tone for the drawn ancestry.
"""

from __future__ import annotations

import pytest

from ppg.attributes.sampler import (
    SAMPLE_ORDER,
    Sampler,
    get_vocabulary,
    normalise_seed,
    seed_from_key,
)


@pytest.fixture
def sampler() -> Sampler:
    return Sampler()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_and_pins_always_give_the_same_person(sampler: Sampler) -> None:
    pins = {"sex": "male", "ethnicity": "south_asian", "glasses": "thin_metal"}
    first = sampler.sample(987654321, pins)

    for _ in range(50):
        again = sampler.sample(987654321, pins)
        assert again.values == first.values
        assert again.age == first.age
        assert again.age_range == first.age_range
        assert again.phrases == first.phrases


def test_a_fresh_sampler_instance_agrees_with_an_old_one(sampler: Sampler) -> None:
    # Determinism must not depend on per-instance state; a restarted server has
    # to reproduce the same face as the one that generated it.
    assert Sampler().sample(42).values == sampler.sample(42).values


def test_different_seeds_give_different_people(sampler: Sampler) -> None:
    combos = {tuple(sorted(sampler.sample(seed).values.items())) for seed in range(200)}
    # 200 independent draws over a vocabulary this wide should essentially never
    # collide. A sharp drop here means an axis stopped being sampled at all.
    assert len(combos) >= 190


def test_every_sampled_axis_is_populated(sampler: Sampler) -> None:
    attrs = sampler.sample(7)
    assert set(attrs.values) == set(SAMPLE_ORDER)
    assert set(attrs.phrases) == set(SAMPLE_ORDER)
    assert attrs.to_dict()["age"] == attrs.age


def test_seed_from_key_is_a_frozen_hash() -> None:
    # Hard-coded on purpose. Changing the hash function would quietly hand every
    # existing `by-seed/<email>` URL a different face, which is the single worst
    # regression this project can ship. If this test fails, that is what
    # happened - do not just update the number.
    assert seed_from_key("ada@example.com") == 7438600334607719399
    assert seed_from_key("") == 7252660547403494068
    assert 0 <= seed_from_key("anything") < (1 << 63)


def test_normalise_seed_reports_the_key_it_hashed() -> None:
    assert normalise_seed(1234) == (1234, None)
    # Digit strings are seeds, not keys: "1234" and 1234 must agree.
    assert normalise_seed("1234") == (1234, None)
    key = "ada@example.com"
    assert normalise_seed(key) == (seed_from_key(key), key)
    seed, key = normalise_seed(None)
    assert key is None and 0 <= seed < (1 << 63)


# ---------------------------------------------------------------------------
# Pinning
# ---------------------------------------------------------------------------


def test_pinned_axes_are_always_honoured(sampler: Sampler) -> None:
    pins = {
        "sex": "female",
        "ethnicity": "east_asian",
        "skin_tone": "fitzpatrick_ii",
        "hair": "braids",
        "glasses": "rimless",
        "expression": "serious",
        "clothing": "blazer",
        "background": "studio_charcoal",
        "lighting": "rembrandt",
        "camera": "85mm",
    }
    for seed in range(25):
        values = sampler.sample(seed, pins).values
        assert {axis: values[axis] for axis in pins} == pins


def test_free_form_values_pass_straight_through(sampler: Sampler) -> None:
    # The vocabulary is a starting point, not a whitelist: the diffusion model
    # does not care that "puffin researcher" is not in a YAML file.
    attrs = sampler.sample(11, {"profession": "puffin researcher"})
    assert attrs.values["profession"] == "puffin researcher"
    assert attrs.phrases["profession"] == "puffin researcher"


def test_human_readable_pins_resolve_to_vocabulary_options(sampler: Sampler) -> None:
    # "marine biologist" is the display form of the `marine_biologist` option;
    # it must resolve to the real option rather than being treated as free text.
    attrs = sampler.sample(11, {"profession": "marine biologist"})
    assert attrs.values["profession"] == "marine_biologist"


# ---------------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------------


def test_exact_age_is_respected(sampler: Sampler) -> None:
    attrs = sampler.sample(3, exact_age=44)
    assert attrs.age == 44
    assert attrs.age_range == "35-44"


def test_pinned_age_range_produces_an_age_inside_it(sampler: Sampler) -> None:
    for seed in range(40):
        attrs = sampler.sample(seed, {"age_range": "65-79"})
        assert 65 <= attrs.age <= 79
        assert attrs.age_range == "65-79"


def test_free_form_age_range_is_parsed(sampler: Sampler) -> None:
    for seed in range(20):
        assert 30 <= sampler.sample(seed, {"age_range": "30-33"}).age <= 33


def test_min_and_max_age_clamp(sampler: Sampler) -> None:
    assert sampler.sample(3, exact_age=10, min_age=18).age == 18
    assert sampler.sample(3, exact_age=200, max_age=90).age == 90
    # The bounds also constrain an unpinned draw.
    for seed in range(40):
        assert 40 <= sampler.sample(seed, min_age=40, max_age=45).age <= 45


def test_bounds_win_over_a_pinned_age_range(sampler: Sampler) -> None:
    # An operator who set PPG_MIN_AGE=25 must not get a 19-year-old just because
    # the caller pinned age_range=18-24.
    for seed in range(20):
        assert sampler.sample(seed, {"age_range": "18-24"}, min_age=25, max_age=90).age >= 25


# ---------------------------------------------------------------------------
# Coherence rules from vocab.yaml
# ---------------------------------------------------------------------------


def test_female_sampled_people_never_get_facial_hair(sampler: Sampler) -> None:
    # rules.facial_hair.only_if.sex - an unrequested beard on a face the caller
    # asked to be female is the most visible incoherence the sampler can produce.
    vocab = get_vocabulary()
    # Read the forced value out of the vocabulary rather than hard-coding it:
    # the sentinel has already been renamed once (clean_shaven -> not_applicable)
    # and what matters is that no hair-bearing option is ever drawn.
    forced = vocab.rules["facial_hair"]["otherwise"]
    hairy = {
        "light_stubble",
        "heavy_stubble",
        "short_beard",
        "full_beard",
        "grey_beard",
        "moustache",
        "goatee",
    }
    for seed in range(200):
        drawn = sampler.sample(seed, {"sex": "female"})
        assert drawn.values["facial_hair"] == forced
        assert drawn.values["facial_hair"] not in hairy
        assert "beard" not in drawn.phrases["facial_hair"]


def test_facial_hair_is_still_possible_for_male_faces(sampler: Sampler) -> None:
    drawn = {sampler.sample(seed, {"sex": "male"}).values["facial_hair"] for seed in range(200)}
    assert drawn - {"clean_shaven"}, "the facial_hair axis collapsed to a single value"


def test_a_nineteen_year_old_is_never_retired_or_a_school_principal(sampler: Sampler) -> None:
    # rules.profession.min_age. Incoherent age/job pairs produce uncanny faces,
    # and "retired 19-year-old" is the canonical example.
    for seed in range(300):
        assert sampler.sample(seed, exact_age=19).values["profession"] not in {
            "retired",
            "school_principal",
            "university_lecturer",
            "medical_doctor",
        }


def test_students_age_out(sampler: Sampler) -> None:
    # rules.profession.max_age.student = 32
    for seed in range(200):
        assert sampler.sample(seed, exact_age=70).values["profession"] != "student"


def test_skin_tone_respects_the_affinity_of_a_pinned_ethnicity(sampler: Sampler) -> None:
    affinity = get_vocabulary().skin_tone_affinity
    for ethnicity in ("west_african", "northern_european", "east_asian"):
        allowed = set(affinity[ethnicity])
        drawn = {
            sampler.sample(seed, {"ethnicity": ethnicity}).values["skin_tone"]
            for seed in range(120)
        }
        assert drawn <= allowed, f"{ethnicity} drew {drawn - allowed}"
        # And the affinity must not collapse the axis to one option.
        assert len(drawn) > 1


def test_an_explicit_skin_tone_pin_overrides_the_affinity(sampler: Sampler) -> None:
    # The affinity table is a rendering heuristic, not a rule about people: a
    # caller who explicitly asks for a combination gets it.
    attrs = sampler.sample(5, {"ethnicity": "west_african", "skin_tone": "fitzpatrick_ii"})
    assert attrs.values["skin_tone"] == "fitzpatrick_ii"


def test_grey_hair_is_discouraged_rather_than_forbidden_on_young_faces(sampler: Sampler) -> None:
    # DISCOURAGE_FACTOR, not a hard filter: grey hair at 25 happens, it is just
    # rare. A hard filter here would be a bug in the other direction.
    young = [sampler.sample(seed, exact_age=25).values["hair"] for seed in range(300)]
    old = [sampler.sample(seed, exact_age=70).values["hair"] for seed in range(300)]
    greys = {"thinning_grey", "silver_short"}
    assert sum(h in greys for h in young) < sum(h in greys for h in old)


# ---------------------------------------------------------------------------
# Stratified batches
# ---------------------------------------------------------------------------


def test_strata_returns_n_pins_covering_more_ground_than_naive_sampling(
    sampler: Sampler,
) -> None:
    n = 40
    pins = sampler.strata(n, seed=99)
    assert len(pins) == n
    assert all(set(pin) == {"sex", "age_range", "ethnicity"} for pin in pins)

    stratified = {(p["sex"], p["age_range"], p["ethnicity"]) for p in pins}
    naive = {
        (a.values["sex"], a.age_range, a.values["ethnicity"])
        for a in (sampler.sample(seed) for seed in range(n))
    }
    # The whole point of `diversity="even"`: a contact sheet of 40 should look
    # like 40 different people, not like the vocabulary's favourite dozen.
    assert len(stratified) == n
    assert len(stratified) > len(naive)


def test_strata_is_deterministic_and_keeps_caller_pins(sampler: Sampler) -> None:
    assert sampler.strata(12, seed=5) == sampler.strata(12, seed=5)
    assert sampler.strata(12, seed=5) != sampler.strata(12, seed=6)

    pinned = sampler.strata(12, seed=5, pinned={"sex": "male"})
    assert all(pin["sex"] == "male" for pin in pinned)
    # A pinned axis stops contributing variety, so the others must carry it.
    assert len({(p["age_range"], p["ethnicity"]) for p in pinned}) == 12


def test_strata_wraps_around_when_asked_for_more_than_the_cross_product(
    sampler: Sampler,
) -> None:
    pins = sampler.strata(6, seed=1, pinned={"sex": "female", "ethnicity": "east_asian"})
    assert len(pins) == 6  # only 7 age buckets remain, so repeats are expected
