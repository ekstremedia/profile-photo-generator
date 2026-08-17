# Prompting and the vocabulary

Everything about what kind of people get generated lives in
`src/ppg/attributes/vocab.yaml`. Editing it needs no Python, and it is the part
of the project most worth contributing to.

## How vocab.yaml is structured

Three top-level sections: `axes`, `rules` and `skin_tone_affinity`, plus a
`version` integer.

### Options

An option is either a plain string or a mapping:

```yaml
axes:
  profession:
    - baker                                     # shorthand
    - { value: hvac_technician, prompt: HVAC technician }
    - { value: lighthouse_keeper, prompt: lighthouse keeper, weight: 0.3 }
  skin_tone:
    - { value: fitzpatrick_iv, label: "IV - olive", prompt: olive tan skin, weight: 1.1 }
  age_range:
    - { value: "25-34", min: 25, max: 34, weight: 1.7 }
```

| Key | Meaning |
| --- | --- |
| `value` | the canonical token. It is what the API accepts, what appears in `attributes`, and what goes into the cache key. Snake case by convention |
| `prompt` | how the option is phrased in the image prompt. Defaults to `value` with underscores replaced by spaces |
| `label` | human-readable name for the gallery form and `/v1/options`. Defaults to the same substitution |
| `weight` | relative sampling weight within the axis, default `1.0`. Weights do not need to sum to anything |
| `min`, `max` | numeric bounds; only meaningful for `age_range` |

Two conventions are load-bearing:

- **`prompt: ""` means "say nothing".** `glasses: none` and
  `clothing: profession_appropriate` both use it. An empty phrase is dropped
  when the prompt is assembled, which leaves the composer free to dress a
  welder like a welder instead of forcing a garment into the sentence.
- **Unknown values pass straight through.** Ask for
  `{"profession": "puffin researcher"}` and the sampler wraps it in an ad-hoc
  option rather than rejecting it — the diffusion model does not care that it
  is not in a YAML file. The human-readable form is also accepted for known
  values, so `"marine biologist"` resolves to `marine_biologist`.

### Sampling order

Axes are drawn in this order, and it matters: axes drawn earlier constrain
those drawn later.

```
sex → age → ethnicity → skin_tone → profession → hair → facial_hair
    → glasses → expression → clothing → background → lighting → camera
```

Age is drawn second because profession, hair and facial hair all depend on it.

### rules

```yaml
rules:
  facial_hair:
    only_if:
      sex: [male, androgynous]
    otherwise: clean_shaven

  hair:
    discourage_below_age:
      thinning_grey: 45
      silver_short: 50
  facial_hair_age:
    discourage_below_age:
      grey_beard: 45

  profession:
    min_age:
      school_principal: 35
      retired: 62
    max_age:
      student: 32
```

- `only_if` / `otherwise` — a hard gate. When the drawn `sex` is not in the
  list, the axis is forced to the `otherwise` value.
- `discourage_below_age` — a soft gate. The option's weight is multiplied by
  `DISCOURAGE_FACTOR` (0.05, in `attributes/sampler.py`) rather than removed.
  Grey hair at 40 happens; it just should not be common.
- `profession.min_age` / `max_age` — a hard filter on the candidate pool.
  Nobody is a school principal at 19.

Rules are ignored for axes the caller pinned explicitly. If you ask for
`{"sex": "female", "facial_hair": "full_beard"}` you get it.

### skin_tone_affinity

```yaml
skin_tone_affinity:
  east_asian: [fitzpatrick_ii, fitzpatrick_iii, fitzpatrick_iv]
  west_african: [fitzpatrick_v, fitzpatrick_vi]
```

When `skin_tone` is not pinned, it is drawn from the subset listed for the
drawn ancestry, using each option's own weight. This is a rendering heuristic
to keep prompts internally coherent — incoherent prompts produce uncanny faces
— and not a claim about any real population. Pinning `skin_tone` bypasses it
entirely.

## Adding or rebalancing entries

Adding a profession is one line:

```yaml
    - { value: bicycle_mechanic, prompt: bicycle mechanic }
```

Then check it:

```bash
ppg options profession                    # every value, weight and phrase
ppg generate --profession bicycle_mechanic --seed demo
```

Guidance from the existing file:

- **Prefer breadth over polish.** The profession list is deliberately full of
  trades, care work, service work and manual work. Avatar sets that are all
  founders and designers look fake immediately.
- **Weight the long tail down, not out.** `lighthouse_keeper` sits at 0.3 so it
  shows up occasionally and never dominates a contact sheet.
- **Keep the phrasing visual.** `prompt` text goes to an image model. "ship's
  engineer" gives it something to draw; "logistics coordination specialist"
  does not.
- **Do not encode correlations you do not mean.** Ancestry, profession, class
  and setting are sampled independently on purpose, and the LLM composer is
  instructed not to infer one from another.
- **Rebalancing weights is a real contribution.** If a batch of 50 comes out
  with twelve people in blazers, the fix is a weight, not code.

Effects on the cache:

- Changing a `weight` changes future draws only; existing avatars are
  untouched.
- Changing a `prompt` string changes the prompt text, which changes the content
  hash, which means new images at new paths. Old ones stay valid.
- Changing something global (the realism cues, the sampler logic) should be
  accompanied by a `PIPELINE_VERSION` bump in `src/ppg/__init__.py`, which
  invalidates the entire store.

Verify a rebalance the same way the project was built:

```bash
ppg batch -n 24 --diversity even --seed rebalance-check
# writes ./out/contact-sheet.jpg
```

A contact sheet is the fastest way to see whether a batch is genuinely varied
or the same face twelve times.

## The realism cue block

Appended to every prompt, LLM or template (`prompt/templates.py`):

```
candid headshot photograph, natural skin texture with visible pores and fine
lines, subtle facial asymmetry, unretouched, sharp focus on the eyes, shallow
depth of field, gentle film grain, colour photograph
```

Each clause is there for a reason. Diffusion models drift towards retouched
stock-photo faces unless you explicitly ask for the imperfections that make a
photograph read as real:

- *visible pores and fine lines*, *unretouched* — the strongest single lever
  against the airbrushed plastic look.
- *subtle facial asymmetry* — real faces are asymmetric; generated ones tend
  towards uncanny symmetry.
- *sharp focus on the eyes*, *shallow depth of field* — how a portrait lens
  actually behaves, and it hides background weaknesses.
- *gentle film grain* — breaks up the too-clean digital surface.
- *colour photograph* — SDXL happily produces black-and-white portraits
  otherwise, which is wrong for an avatar set.
- *candid headshot photograph* — "photograph" as the medium, up front, where
  SDXL weights it heavily.

The framing clause is kept separate so `minor_mode` can swap it:

```
FRAMING       head and shoulders portrait, centred, looking at the camera
FRAMING_PLAIN plain head and shoulders school portrait, centred, looking at the
              camera, neutral
```

## The base negative prompt

```
cartoon, anime, illustration, painting, drawing, sketch, 3d render, cgi, video
game character, doll, mannequin, plastic skin, waxy skin, airbrushed,
oversaturated, overexposed, heavy makeup, beauty filter, instagram filter,
deformed, disfigured, face asymmetry, deformed eyes, crossed eyes, extra
fingers, extra limbs, bad anatomy, bad proportions, long neck, watermark,
signature, text, logo, username, blurry, out of focus, low quality, jpeg
artifacts, grainy noise, black and white, monochrome
```

Four groups: the illustration/CGI basin (SDXL's training data is full of it),
the over-processed beauty look, the usual anatomy failures, and the artefacts —
watermarks and signatures in particular, which SDXL adds unprompted because so
much of its training data had them.

Two apparent contradictions are deliberate. *face asymmetry* in the negative
alongside *subtle facial asymmetry* in the positive: the negative pushes away
from the distorted, sliding-features failure, the positive asks for the small
natural kind. Likewise *grainy noise* against *gentle film grain*.

`negative_extra` is appended to this rather than replacing it, and it is only
length-checked, not word-filtered — blocking terms in a negative prompt would
be exactly backwards.

## Tuning steps and guidance

Defaults, in `.env.example`:

```
PPG_STEPS=30
PPG_FAST_STEPS=15
PPG_GUIDANCE=4.5
```

**Guidance is the one to get right.** RealVisXL V5 wants CFG around 4–5. The SD
1.5-era habit of 7–8 is the main cause of the over-saturated, over-contrasted,
waxy "AI face": high CFG forces the model onto every token hard, which
exaggerates the skin, the colours and the expression all at once. If your
output looks like a magazine cover rather than a work photo, lower the
guidance before changing anything else.

- 3.0–4.0: softer, more natural, occasionally ignores an attribute.
- 4.5: the default. Attributes land, skin stays believable.
- 6.0+: visibly over-processed on this model. Avoid.

**Steps** are a much flatter curve. 30 is comfortably converged for DPM++ 2M
SDE Karras; 15 (`fast: true`) is slightly softer and roughly halves the time,
which is a good trade for a 128 px avatar and a poor one for a 1024 px
download. Above about 40 steps the differences are hard to see.

**Prompt length.** CLIP truncates at 77 tokens, and a fully populated prompt
plus the realism block goes past that — diffusers logs a warning saying so.
The order in `templates.py` puts framing and subject first for that reason, so
what gets truncated is the tail of the scene description rather than the
person. Keep `prompt_extra` short, and if you add to the realism block, remove
something else.
