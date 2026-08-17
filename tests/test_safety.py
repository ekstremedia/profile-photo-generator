"""The content rules.

Free text is the only untrusted input this service takes, so this module is the
whole attack surface: everything else is drawn from a curated vocabulary. The
rules are deliberately narrow and auditable - sexual content, age-through-text,
and attempts to steer at a real person.
"""

from __future__ import annotations

import pytest

from ppg.config import Settings
from ppg.safety import SafetyError, check_free_text, clamp_age, minor_mode

# ---------------------------------------------------------------------------
# Blocked free text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "nude",
        "wearing lingerie",
        "a sexy pose",
        "NSFW please",
        "topless portrait",
    ],
)
def test_sexual_terms_are_refused(text: str) -> None:
    with pytest.raises(SafetyError) as excinfo:
        check_free_text(text)
    assert "sexual" in str(excinfo.value)


@pytest.mark.parametrize(
    "text",
    [
        "a child",
        "teenager with a backpack",
        "schoolgirl uniform",
        # Multi-word phrases take the substring branch of _matched(), which is a
        # separate code path from the whole-word token match above.
        "young girl smiling",
        "little boy",
        "16 year old",
    ],
)
def test_minor_terms_are_refused(text: str) -> None:
    with pytest.raises(SafetyError) as excinfo:
        check_free_text(text)
    assert "age" in str(excinfo.value)


@pytest.mark.parametrize(
    "text",
    [
        "famous",
        "celebrity smile",
        # Again the multi-word path, and the hyphenated variant.
        "face swap of a colleague",
        "looks like a well-known actor",
        "look-alike",
    ],
)
def test_real_person_markers_are_refused(text: str) -> None:
    with pytest.raises(SafetyError) as excinfo:
        check_free_text(text)
    assert "do not exist" in str(excinfo.value)


def test_the_blocked_term_is_named_in_the_error() -> None:
    # A refusal that does not say what was wrong is a support ticket.
    with pytest.raises(SafetyError, match="nude"):
        check_free_text("nude")


def test_the_field_name_is_named_in_the_error() -> None:
    with pytest.raises(SafetyError, match="negative_extra"):
        check_free_text("nude", field="negative_extra")


def test_substrings_of_innocent_words_are_not_blocked() -> None:
    # Single words match on token boundaries, so "kidney" must not trip "kid"
    # and "screenwriter" must not trip "teen".
    assert check_free_text("kidney specialist") == "kidney specialist"
    assert check_free_text("screenwriter at a desk") == "screenwriter at a desk"


# ---------------------------------------------------------------------------
# Accepted free text
# ---------------------------------------------------------------------------


def test_clean_text_passes_and_is_whitespace_normalised() -> None:
    assert check_free_text("  wearing   a  wool \n scarf  ") == "wearing a wool scarf"
    assert check_free_text(None) == ""
    assert check_free_text("") == ""


def test_text_over_500_characters_is_refused() -> None:
    assert len(check_free_text("a " * 250)) == 499  # just inside the limit
    with pytest.raises(SafetyError, match="too long"):
        check_free_text("x" * 501)


# ---------------------------------------------------------------------------
# Age clamping
# ---------------------------------------------------------------------------


def test_clamp_age_passes_through_and_clamps_at_the_top(settings: Settings) -> None:
    assert clamp_age(None, settings) is None
    assert clamp_age(30, settings) == 30
    assert clamp_age(200, settings) == settings.max_age


def test_clamp_age_refuses_below_the_minimum_when_minors_are_not_allowed(
    settings: Settings,
) -> None:
    assert settings.allow_minors is False
    with pytest.raises(SafetyError) as excinfo:
        clamp_age(12, settings)
    # The refusal has to explain the opt-in, otherwise operators just guess.
    assert "PPG_ALLOW_MINORS" in str(excinfo.value)


def test_clamp_age_clamps_instead_of_raising_when_minors_are_allowed(
    settings: Settings,
) -> None:
    permissive = settings.model_copy(update={"allow_minors": True})
    assert clamp_age(12, permissive) == 12
    assert clamp_age(-5, permissive) == 0  # floor drops to 0, nothing raises
    assert clamp_age(200, permissive) == permissive.max_age


def test_minor_mode_only_engages_under_the_opt_in(settings: Settings) -> None:
    assert minor_mode(12, settings) is False  # never, without the opt-in
    permissive = settings.model_copy(update={"allow_minors": True})
    assert minor_mode(12, permissive) is True
    assert minor_mode(30, permissive) is False


# ---------------------------------------------------------------------------
# The same rules, through the API
# ---------------------------------------------------------------------------


def test_a_blocked_request_is_a_422_with_a_helpful_detail(client) -> None:
    response = client.post("/v1/avatars", json={"prompt_extra": "nude", "seed": 1})
    # 422 rather than 400: it is a semantically invalid request, and the
    # SafetyError handler in create_app() is what turns it into one.
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "prompt_extra" in detail
    assert "nude" in detail


def test_a_blocked_age_is_a_422(client) -> None:
    response = client.post("/v1/avatars", json={"age": 9, "seed": 1})
    assert response.status_code == 422
    assert "PPG_ALLOW_MINORS" in response.json()["detail"]


def test_the_negative_prompt_is_not_content_filtered(client) -> None:
    # "nude" in a negative prompt is an instruction to avoid nudity, so
    # blocking it there would be backwards.
    response = client.post("/v1/avatars", json={"negative_extra": "nude", "seed": 2})
    assert response.status_code == 200
    assert "nude" in response.json()["negative_prompt"]
