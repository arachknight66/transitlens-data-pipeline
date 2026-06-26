# real_tess/ — Real TESS Data Integration (Stretch Goal)

> **Status:** Stretch goal (Tier 3). The offline synthetic demo (`synthetic/`) does
> not depend on anything in this folder. Everything here is an enhancement, not
> a dependency — if MAST or the network is unavailable, `interface.py` still
> works perfectly for `source="synthetic"`.

## What this does

Loads a real TESS light curve for a given TIC ID from
[MAST](https://mast.stsci.edu/) via [Lightkurve](https://docs.lightkurve.org/),
normalises it to this pipeline's standard `(time, flux)` shape, and caches the
raw `.fits` file locally so repeat calls don't re-hit the network.

| File | Responsibility | Talks to the network? |
|---|---|---|
| `mast_loader.py` | Search, download, cache `.fits` files | Yes — the only file that does |
| `sector_selector.py` | Pick the best sector from search metadata | No — pure function |
| `flux_normaliser.py` | PDC-SAP flux → normalised flux (median ≈ 1.0) | No — pure function |
| `cache/` | On-disk store of downloaded `.fits` files (gitignored except `.gitkeep`) | — |

## Enabling real TESS data

This pipeline ships with `lightkurve`/`astroquery` **commented out** in
`requirements.txt` so the offline hackathon demo installs in under a minute.
To enable Phase 5:

```bash
# uncomment lightkurve and astroquery in requirements.txt, then:
pip install -r requirements.txt
```

After that, `interface.py` will resolve `source="tess"` automatically —
no other code changes needed.

## Verified demo targets

These three TIC IDs are recommended for hackathon demos because they have
bright, well-studied, clean signals:

| TIC ID | Planet | Period (days) | Depth | Why good for demo |
|---|---|---|---|---|
| `25155310` | WASP-126 b | 3.29 | 0.011 | Bright star, clean signal, extensively studied |
| `279741377` | TOI-270 b | 3.36 | 0.005 | Multi-planet system, community favourite |
| `149603524` | LHS 3844 b | 0.46 | 0.004 | Ultra-short period, many transits per sector |

If you have internet access before the event, pre-populate the cache so the
live demo never depends on venue WiFi:

```python
from real_tess.mast_loader import fetch_light_curve

for tic_id in ["25155310", "279741377", "149603524"]:
    fetch_light_curve(tic_id, cache_dir="real_tess/cache")
```

## Cache file naming convention

```
real_tess/cache/TIC{tic_id}_sector{sector:03d}.fits
```

e.g. `real_tess/cache/TIC25155310_sector015.fits`

`mast_loader.fetch_light_curve()` checks this cache *before* attempting any
network call, and writes back to it after every fresh download — so once a
target has been fetched once (on any machine, as long as the cache file is
copied over), subsequent calls are fully offline.

## Error handling

| Situation | Behaviour |
|---|---|
| `lightkurve` not installed | Raises `ImportError` with install instructions |
| No observations for the TIC ID | Raises `real_tess.mast_loader.TessDataUnavailableError` |
| Network unreachable, nothing cached | Raises `TessDataUnavailableError` |
| Network unreachable, cache hit | Returns cached data silently — no error |
| Download times out | Retries once, then raises `TessDataUnavailableError` |

## Known limitations

- Real TESS sectors have a ~1 day data-downlink gap near the sector midpoint.
  `flux_normaliser.py` does **not** fill this gap — `interface.py`'s
  `_load_tess()` returns the gapped time array as-is, and `ml-core`'s BLS
  implementation must handle gapped arrays correctly (it already does, since
  the synthetic generator also injects ~2% random gaps).
- `label`, `true_period`, `true_depth`, and `true_duration` are `None` for
  real TESS targets unless explicitly passed in via `config` — this pipeline
  does not infer ground truth from MAST. See `interface.py`'s `_load_tess()`
  for the optional override keys.

## Sector Ingestion and Screening (Phase 2)

Phase 2 introduces automated ingestion and screening of full/sub-sectors:
* `sector_manifest.py`: Builds a manifest of observations matching a TESS sector and cadence from MAST (or local cache files).
* `sector_downloader.py`: Downloads pending FITS files from MAST using Lightkurve (prioritizing 120-second cadence).
* `sector_processor.py`: Parses downloaded FITS files and outputs Phase 1-compliant `.npz` arrays (cleaning NaN/Inf/quality-flag points, normalising by median).
* `sector_screening.py`: Iterates over the preprocessed arrays, runs the detection and classification models, and generates summary CSVs and a screening report.

### Run Screening Workflow
From `transitlens-ml-core`:
```bash
python -m eval.run_sector_screening --sector 98 --limit 100 --download
```
Outputs are written to `transitlens-data-pipeline/datasets/processed/tess_sector/`.