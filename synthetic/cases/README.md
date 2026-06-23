# synthetic/cases/ — Generated Light Curves

This folder holds the CSV output of `synthetic/generator.py`. **Nothing here is
committed by hand** — every file is reproducible from `synthetic/config.yaml`
by running:

```python
from synthetic.generator import generate_all_cases
generate_all_cases("synthetic/config.yaml", "synthetic/cases")
```

`tests/conftest.py` also runs this automatically on a fresh checkout if these
CSVs don't exist yet, so you never *have* to run it manually before `pytest`.

Each CSV has exactly two columns, `time` (BTJD) and `flux` (normalised,
median ≈ 1.0) — see the output contract in `interface.py` and the column
definitions in `datasets/schema.md` for the full schema once these are
merged into `labeled_dataset.csv`.

---

## candidate_a — `exoplanet_like`

A shallow, flat-bottomed, strictly periodic transit — the signature of a
Jupiter-sized planet transiting a Sun-like star.

| Parameter | Value | Why |
|---|---|---|
| `period_days` | 3.42 | Typical hot/warm-Jupiter period |
| `depth` | 0.013 (1.3%) | Consistent with a Jupiter-radius planet blocking ~1.3% of the stellar disc |
| `duration_days` | 0.16 (~3.8 hours) | Realistic transit duration for this period/radius combination |
| `v_shape` | `false` | Box-shaped dip — flat bottom, sharp ingress/egress |
| `secondary_eclipse` | `false` | No secondary dip; a planet doesn't occult enough starlight to register one |
| `noise_type` / `noise_level` | gaussian / 0.002 | 2000 ppm white noise, realistic for a moderately bright (~V=11) star |
| `seed` | 42 | Reproducibility |

**What ml-core should recover:** a BLS search should find a strong peak at
period ≈ 3.42 days with a flat-bottomed, box-shaped folded transit.

## candidate_b — `eclipsing_binary_like`

A deep, V-shaped primary eclipse with a secondary eclipse at half-period —
the signature of two stars of comparable size occulting each other.

| Parameter | Value | Why |
|---|---|---|
| `period_days` | 1.87 | Short period typical of close binaries |
| `depth` | 0.18 (18%) | Far too deep for any planet — immediately signals a stellar-mass companion |
| `duration_days` | 0.08 | Shorter eclipse duration consistent with the tighter orbit |
| `v_shape` | `true` | Triangular profile — the secondary star has finite size relative to the primary, producing a curved ingress/egress rather than a flat bottom |
| `secondary_eclipse` | `true`, depth 0.09 | The secondary star passing behind the primary; roughly half the primary depth |
| `noise_type` / `noise_level` | gaussian / 0.003 | Slightly noisier than candidate_a |
| `seed` | 43 | Reproducibility |

**What ml-core should recover:** a BLS search should find a peak at period ≈
1.87 days, but the folded profile should be V-shaped (not flat-bottomed) and
a second, shallower dip should appear at phase 0.5 — these two features are
what should drive a classifier to label this `eclipsing_binary_like` rather
than `exoplanet_like`.

## candidate_c — `noise_or_other`

No injected transit at all — red (correlated) noise plus a sinusoidal
stellar-variability signal, deliberately chosen to have real structure
without ever being truly periodic on a transit-like timescale.

| Parameter | Value | Why |
|---|---|---|
| `period_days` / `depth` / `duration_days` | `null` | No transit is injected |
| `noise_type` | red | AR(1)-correlated noise, mimicking instrumental systematics |
| `noise_level` | 0.015 | Deliberately noisy — this is meant to be a hard "is there anything here?" case |
| (stellar variability) | amplitude 0.005, period 12 days | Added automatically for `noise_type: red` cases in `generate_from_config()`, simulating starspot rotation |
| `seed` | 44 | Reproducibility |

**What ml-core should recover:** no significant BLS peak above the
detection threshold. If a peak does appear, it should be far weaker than the
peaks recovered for candidate_a/b — this case exists specifically to check
the classifier doesn't have a high false-positive rate on noisy-but-flat
light curves.

---

## Adding a new case

See `CONTRIBUTING.md`'s "How to add a new synthetic case" section — in
short: add an entry to `synthetic/config.yaml`, regenerate, re-run
`datasets/build_dataset.py`, and document the new case's physical scenario
here following the table format above.