"""Prompt composition.

The template composer is the default path and the fallback path, so it has to
be complete on its own: no Ollama, no network, no surprises. The LLM path is
exercised here only through stubs - a test suite that needed a running Ollama
would not be a test suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from ppg.attributes.sampler import Attributes, Sampler
from ppg.prompt.composer import (
    PREFERRED_MODELS,
    AutoComposer,
    LLMComposer,
    TemplateComposer,
    _names_data,
    resolve_ollama_model,
)
from ppg.prompt.ollama_client import OllamaClient, OllamaError
from ppg.prompt.templates import (
    FRAMING,
    FRAMING_PLAIN,
    NEGATIVE_SUBJECT,
    NEGATIVE_WORD_BUDGET,
    PROMPT_WORD_BUDGET,
    REALISM_CUES,
)

PINS = {
    "sex": "female",
    "ethnicity": "east_asian",
    "profession": "marine_biologist",
    "glasses": "thin_metal",
    "hair": "long_straight",
    "expression": "warm_smile",
}


@pytest.fixture
def attrs() -> Attributes:
    return Sampler().sample(20250817, PINS, exact_age=37)


# ---------------------------------------------------------------------------
# Template composer
# ---------------------------------------------------------------------------


async def test_the_template_prompt_carries_the_realism_cues_and_the_pinned_traits(
    attrs: Attributes,
) -> None:
    result = await TemplateComposer().compose(attrs, seed=20250817)

    assert result.source == "template"
    # The realism block and the framing are appended by us, never left to a
    # model: they are what make the output read as a photograph.
    assert FRAMING in result.prompt
    assert REALISM_CUES in result.prompt

    for phrase in (
        "37 year old",
        "East Asian",
        "female",
        "thin metal-framed glasses",
        "long straight hair",
        "warm genuine smile",
    ):
        assert phrase in result.prompt, f"{phrase!r} missing from the prompt"

    assert NEGATIVE_SUBJECT in result.negative_prompt


async def test_the_prompt_stays_inside_the_token_budget(attrs: Attributes) -> None:
    """CLIP discards the tail beyond 77 tokens, so the prompt must stay short.

    Both of SDXL's text encoders receive this same string - splitting subject
    and style across the two encoders looks like free capacity but measurably
    loses requested details, because the pooled conditioning comes from the
    second encoder alone.
    """
    result = await TemplateComposer().compose(attrs, seed=1)

    assert len(result.prompt.split()) <= PROMPT_WORD_BUDGET
    assert len(result.negative_prompt.split()) <= NEGATIVE_WORD_BUDGET


async def test_what_the_caller_asked_for_survives_the_budget(attrs: Attributes) -> None:
    """Requested content outranks generated filler when the budget is tight.

    This is the regression that mattered in practice: a request for a red scarf
    used to sit at the very end of the prompt, behind forty words of invented
    detail, and fell off the 77-token cliff without a word of warning.
    """
    result = await TemplateComposer().compose(
        attrs,
        seed=1,
        extra="wearing a bright red scarf",
        negative_extra="hats",
        plain_framing=True,
    )

    assert "wearing a bright red scarf" in result.prompt
    assert FRAMING_PLAIN in result.prompt
    assert "hats" in result.negative_prompt
    # Pinned attributes are caller intent too, so they are protected as well.
    assert "thin metal-framed glasses" in result.prompt
    # And the cues that make it photorealistic are never sacrificed either.
    assert REALISM_CUES in result.prompt


async def test_an_elderly_age_gets_explicit_appearance_cues(attrs: Attributes) -> None:
    """A bare "90 year old" renders as a well-preserved sixty.

    SDXL's training data skews heavily to attractive thirty-somethings, so the
    visible consequences of age have to be stated positively and youth pushed
    away in the negative prompt.
    """
    old = Sampler().sample(seed=7, pinned={"sex": "male"}, exact_age=90)
    result = await TemplateComposer().compose(old, seed=7)

    assert "90 year old" in result.prompt
    assert "very elderly" in result.prompt
    assert "wrinkled" in result.prompt
    assert "youthful" in result.negative_prompt

    young = Sampler().sample(seed=7, pinned={"sex": "male"}, exact_age=22)
    young_result = await TemplateComposer().compose(young, seed=7)
    assert "elderly" in young_result.negative_prompt


async def test_absent_features_are_pushed_into_the_negative_prompt(attrs: Attributes) -> None:
    """ "No glasses" has no useful positive phrasing.

    Writing "no glasses" into a prompt is as likely to summon glasses as to
    prevent them, so options describing an absence carry negative terms in
    vocab.yaml instead.
    """
    bare = Sampler().sample(seed=3, pinned={"glasses": "none", "facial_hair": "clean_shaven"})
    result = await TemplateComposer().compose(bare, seed=3)

    assert "glasses" in result.negative_prompt
    assert "beard" in result.negative_prompt


async def test_composing_twice_with_the_same_seed_is_identical(attrs: Attributes) -> None:
    first = await TemplateComposer().compose(attrs, seed=555)
    second = await TemplateComposer().compose(attrs, seed=555)

    assert first.prompt == second.prompt
    assert first.negative_prompt == second.negative_prompt
    # The persona is drawn from a name pool with its own RNG, so it is the part
    # most likely to drift if the seeding is ever broken.
    assert first.persona == second.persona

    different = await TemplateComposer().compose(attrs, seed=556)
    assert different.persona != first.persona


async def test_the_persona_name_comes_from_the_pool_for_the_sampled_ancestry(
    attrs: Attributes,
) -> None:
    result = await TemplateComposer().compose(attrs, seed=20250817)
    persona = result.persona
    assert persona is not None

    data = _names_data()
    pool = data["pools"][data["map"]["east_asian"]]
    given, family = persona.name.rsplit(" ", 1)
    # A Nordic surname on an East Asian face is exactly the kind of incoherence
    # the ethnicity -> pool mapping exists to prevent.
    assert given in pool["female"]
    assert family in pool["family"]
    assert persona.city in pool["cities"]

    assert persona.age == attrs.age  # never the model's or the pool's own idea
    assert persona.occupation == "marine biologist"


async def test_the_persona_falls_back_to_a_pool_for_unmapped_ancestries() -> None:
    attrs = Sampler().sample(3, {"ethnicity": "puffin islander"})
    persona = await TemplateComposer().compose(attrs, seed=3)
    assert persona.persona is not None  # free-form ancestry must not crash naming


# ---------------------------------------------------------------------------
# Auto composer fallback
# ---------------------------------------------------------------------------


class _FailingOllama:
    """Stands in for OllamaClient. Never opens a socket."""

    def __init__(self, error: str = "connection refused") -> None:
        self.error = error
        self.calls = 0

    async def chat_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raise OllamaError(self.error)


async def test_auto_composer_falls_back_to_the_template_when_ollama_fails(
    attrs: Attributes,
) -> None:
    client = _FailingOllama()
    auto = AutoComposer(LLMComposer(client), TemplateComposer())  # type: ignore[arg-type]

    result = await auto.compose(attrs, seed=42)

    # A machine without Ollama must still generate avatars: "no Ollama" is a
    # supported configuration, not an outage.
    assert result.source == "template"
    assert REALISM_CUES in result.prompt
    assert result.persona is not None
    # Two attempts before giving up - a transient failure should not cost the
    # richer wording immediately.
    assert client.calls == 2

    expected = await TemplateComposer().compose(attrs, seed=42)
    assert result.prompt == expected.prompt


async def test_the_llm_composer_itself_still_raises(attrs: Attributes) -> None:
    # Only AutoComposer swallows the error; PPG_COMPOSER=llm is an explicit
    # request for the LLM and should fail loudly rather than pretend.
    with pytest.raises(OllamaError):
        await LLMComposer(_FailingOllama()).compose(attrs, seed=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


class _StubOllama:
    def __init__(self, models: list[dict[str, Any]] | None = None, fail: bool = False) -> None:
        self.models = models or []
        self.fail = fail

    async def installed_models(self) -> list[dict[str, Any]]:
        if self.fail:
            raise OllamaError("no Ollama here")
        return self.models


def _model(name: str, size: int = 1 << 30) -> dict[str, Any]:
    return {"name": name, "size": size}


async def test_an_explicit_model_is_used_verbatim() -> None:
    client = _StubOllama([_model("llama3.2:1b")])
    assert await resolve_ollama_model(client, "my-own:latest") == "my-own:latest"  # type: ignore[arg-type]


async def test_auto_prefers_the_smallest_known_model() -> None:
    client = _StubOllama([_model("llama3.1:8b"), _model("llama3.2:3b"), _model("mistral:7b")])
    # PREFERRED_MODELS is ordered by size/reliability, not by what is installed.
    assert await resolve_ollama_model(client, "auto") == "llama3.2:3b"  # type: ignore[arg-type]


async def test_auto_never_substitutes_qwen3_8b_for_qwen3_4b() -> None:
    # Regression guard for a real bug: matching on the bare family name made a
    # request for `qwen3:4b` select `qwen3:8b`, twice the VRAM and enough to
    # push the diffusion model out of a 12GB card.
    client = _StubOllama([_model("qwen3:8b"), _model("llama3.2:1b")])
    chosen = await resolve_ollama_model(client, "auto")  # type: ignore[arg-type]

    assert chosen != "qwen3:8b"
    assert chosen == "llama3.2:1b"
    assert PREFERRED_MODELS.index("qwen3:4b") < PREFERRED_MODELS.index("qwen3:8b")


async def test_auto_tolerates_quantisation_suffixes_on_the_same_tag() -> None:
    # "llama3.2:3b-instruct-q4_K_M" is the same model at the same size, so it
    # is a legitimate match for "llama3.2:3b" - unlike qwen3:8b above.
    client = _StubOllama([_model("llama3.2:3b-instruct-q4_K_M"), _model("qwen3:8b")])
    assert await resolve_ollama_model(client, "auto") == "llama3.2:3b-instruct-q4_K_M"  # type: ignore[arg-type]


async def test_auto_falls_back_to_the_smallest_installed_model() -> None:
    client = _StubOllama(
        [_model("exotic:70b", size=40 << 30), _model("homegrown:2b", size=2 << 30)]
    )
    assert await resolve_ollama_model(client, "auto") == "homegrown:2b"  # type: ignore[arg-type]


@pytest.mark.parametrize("client", [_StubOllama([]), _StubOllama(fail=True)])
async def test_no_models_or_no_ollama_resolves_to_none(client: _StubOllama) -> None:
    # None is how the caller learns to use the template composer instead; an
    # exception here would take the whole service down at startup.
    assert await resolve_ollama_model(client, "auto") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Ollama client details
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("0", 0),
        ("30", 30),
        ("5m", "5m"),
        ("-1", -1),
    ],
)
def test_keep_alive_digits_are_sent_as_a_number(configured: str, expected: int | str) -> None:
    # Regression guard for a real bug: Ollama ignores the string "0" because it
    # is not a valid duration, leaving the model resident for the default five
    # minutes - which is exactly the VRAM the diffusion model then cannot get.
    value = OllamaClient("http://localhost:11434", "m", keep_alive=configured)._keep_alive_value()
    assert value == expected
    assert isinstance(value, type(expected))


def test_the_ollama_base_url_is_normalised() -> None:
    client = OllamaClient("http://localhost:11434/", "m")
    assert client.base_url == "http://localhost:11434"
