# transitlens-data-pipeline

> **Bharatiya Antariksh Hackathon 2026 — PS7**  
> Feeds cleaned, labelled light curve data to `transitlens-ml-core`

---

## What this repo does

`transitlens-data-pipeline` has exactly one job: **produce light curve data in a standard shape that `transitlens-ml-core` can consume.**

It handles:
- Generating synthetic TESS-like light curves (MVP — offline, no internet required)
- Loading and caching real TESS data from MAST via Lightkurve (stretch)
- Assembling labelled datasets for training and evaluation
- Exposing a single entry-point function `load_light_curve()` via `interface.py`

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify install
python -c "import numpy, pandas, scipy, yaml; print('OK')"

# 3. Generate synthetic light curves
python -c "
from synthetic.generator import generate_all_cases
generate_all_cases('synthetic/config.yaml', 'synthetic/cases')
"

# 4. Load a light curve via interface
python -c "
from interface import load_light_curve
result = load_light_curve('synthetic', 'candidate_a')
print(f'Loaded {result[\"n_points\"]} points for {result[\"target_id\"]}')
"
```

## Folder structure

```
transitlens-data-pipeline/
│
├── README.md                          ← this file
├── CONTRIBUTING.md                    ← how to add new synthetic cases
├── requirements.txt                   ← numpy, pandas, scipy, pyyaml (+ optional lightkurve)
├── .gitignore                         ← ignore cache/, *.fits, *.pyc, __pycache__
├── interface.py                       ← load_light_curve() — only file ml-core imports
│
├── synthetic/
│   ├── __init__.py
│   ├── generator.py                   ← base time array + flux generation
│   ├── noise_models.py                ← Gaussian + red noise injection
│   ├── transit_injector.py            ← periodic dip injection
│   ├── config.yaml                    ← all case parameters
│   └── cases/
│       ├── candidate_a.csv            ← exoplanet_like output
│       ├── candidate_b.csv            ← eclipsing_binary_like output
│       └── candidate_c.csv            ← noise_or_other output
│
├── real_tess/                         ← stretch: real TESS data loading
├── datasets/                          ← assembled labelled datasets
├── notebooks/                         ← visual exploration
└── tests/                             ← pytest test suite
```

## Output contract

Every call to `load_light_curve()` returns this exact shape:

```python
{
    "time":      List[float],     # BTJD timestamps
    "flux":      List[float],     # normalised flux, median ~1.0
    "target_id": str,             # e.g. "candidate_a"
    "source":    str,             # "synthetic" | "tess" | "csv"
    "n_points":  int,             # len(time) == len(flux)
    "metadata": {
        "cadence_min":    float,
        "time_span_days": float,
        "sector":         int | None,
        "label":          str | None,
        "true_period":    float | None,
        "true_depth":     float | None,
        "true_duration":  float | None,
    }
}
```

## Synthetic cases

| Case | Label | Description |
|---|---|---|
| `candidate_a` | `exoplanet_like` | Shallow periodic transit, flat bottom, 1.3% depth |
| `candidate_b` | `eclipsing_binary_like` | Deep V-shaped primary + secondary eclipse, 18% depth |
| `candidate_c` | `noise_or_other` | No transit — red noise + stellar variability |

## License

Hackathon project — Bharatiya Antariksh Hackathon 2026
