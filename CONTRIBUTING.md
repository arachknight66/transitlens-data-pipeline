# Contributing to transitlens-data-pipeline

This repo's only job is producing correctly-shaped, labelled light curve data
for `transitlens-ml-core` to consume. Keep that boundary in mind for every
change — see [The boundary rule](#the-boundary-rule) below before adding
anything that feels like it might belong somewhere else.

---

## Coding conventions

- **Functions and variables:** `snake_case` (e.g. `inject_transit`, `cadence_minutes`).
- **Classes:** `PascalCase` (e.g. `TessDataUnavailableError`).
- **Module-private helpers:** prefix with a single underscore (e.g. `_load_synthetic`,
  `_wrapped_phase`) so it's obvious at a glance what's internal vs. part of a
  module's public API.
- **Docstrings:** every public function gets a NumPy-style docstring
  (`Parameters` / `Returns` / `Raises` sections) — match the style already used
  throughout `synthetic/`, `real_tess/`, and `interface.py`.
- **No hardcoded parameters.** Generation parameters belong in
  `synthetic/config.yaml`, not scattered through code. If you find yourself
  writing a literal period/depth/noise value inside a function body (outside
  of a docstring example or a test), it should be a config key or a function
  argument instead.
- **Vectorise over numpy arrays.** Don't write `for i in range(len(time)): ...`
  loops over light curve data — every existing generator/injector/normaliser
  uses numpy boolean masking or broadcasting instead, both for performance
  (10x+ faster at TESS-cadence array sizes) and because it's the established
  style here. If a computation is genuinely sequential (e.g. an AR(1)
  recursion), use `scipy.signal.lfilter` rather than a raw Python loop — see
  `synthetic/noise_models.py:add_red_noise` for the pattern.

---

## How to add a new noise model

1. Add a new function to `synthetic/noise_models.py` following the existing
   signature pattern: `add_x_noise(flux, ..., seed=None) -> np.ndarray`. It
   must not mutate `flux` in place (always work on a copy or a fresh array)
   and must accept a `seed` for reproducibility.
2. Wire it into `synthetic/generator.py`'s `generate_from_config()` by adding
   a new branch to the `noise_type` dispatch (currently `'gaussian'` / `'red'`).
3. Add the new `noise_type` string as a valid value for the `noise_type` key
   in `synthetic/config.yaml` cases, and document it in this file's "Valid
   `noise_type` values" if you add one permanently.
4. Write tests in `tests/test_noise_models.py` covering: the noise has the
   expected statistical property (mean, std, autocorrelation, etc. — whatever
   is specific to your model), and same-seed reproducibility / different-seed
   divergence (see `test_seed_reproducibility` for the pattern every existing
   noise function follows).

## How to add a new target class

The three existing classes (`exoplanet_like`, `eclipsing_binary_like`,
`noise_or_other`) are the full label vocabulary `transitlens-ml-core` and
`datasets/schema.md` currently expect. Adding a fourth class is a breaking
change to both repos' contract, so:

1. Confirm with whoever owns `transitlens-ml-core` before doing this — the
   classifier's label set has to change in lockstep.
2. Add the new label string to `_VALID_LABELS` in `interface.py` and to the
   "Valid `label` values" table in `datasets/schema.md`.
3. Add at least one new synthetic case in `synthetic/config.yaml` under
   `cases:` demonstrating the new class (see Phase 5.4 in the build plan for
   an example — stellar variability mimicking a transit is the next natural
   candidate class).
4. Update `tests/test_interface.py`'s `test_candidate_labels_match_expected`
   and `tests/test_schema.py`'s `VALID_LABELS` constant.

## Testing requirements

- **Every new function needs a test.** This includes private helper
  functions if they contain any non-trivial logic (e.g. `_wrapped_phase`,
  `_normalise_tic_id`) — a test for a one-line passthrough wrapper isn't
  required, but anything with a branch or a calculation is.
- Tests live in `tests/`, one file per module being tested
  (`test_generator.py` ↔ `synthetic/generator.py`, etc.) — follow that
  naming convention for new modules.
- Run the full suite before opening a PR: `pytest tests/ -v`. It must pass
  with zero failures.
- Prefer testing with small, fast synthetic inputs (the `synthetic_time` /
  `synthetic_flux` fixtures in `conftest.py`, or hand-built arrays) over
  always running the full 18,000-point generation pipeline — keep the suite
  fast.
- If your change touches anything that affects `labeled_dataset.csv`'s
  shape, re-check `tests/test_schema.py`'s validation rules still hold; if
  it touches `load_light_curve()`'s return shape, re-check
  `tests/test_interface.py`'s contract tests.

---

## The boundary rule

This repo produces data. It does not analyze it. Concretely:

| Belongs here (`data-pipeline`) | Belongs in `ml-core` |
|---|---|
| Generating/loading/caching light curves | Running BLS or any transit-detection algorithm |
| Injecting noise and transit signals | Classifying signals or assigning confidence scores |
| Assembling labelled datasets, train/val/test splits | Computing precision/recall/F1 on detections |
| Validating the `load_light_curve()` output contract | Rendering plots, dashboards, or HTTP endpoints |

If you're writing code that inspects a light curve's *shape* to decide
something about its astrophysical nature (is this periodic? is this a
transit?), that's `ml-core`'s job, not this repo's. This repo's only opinion
about a light curve's content is whatever ground-truth label it was
generated with (synthetic) or explicitly told about via `config` (real TESS).

This repo must never import from `transitlens-ml-core` or
`transitlens-platform`. If you find yourself wanting to, that's a sign the
function belongs in one of those repos instead.