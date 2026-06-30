# HANDOFF.md

# TransitLens Data Pipeline Handoff

## Repository Status

Current Phase

Phase 7 complete - Testing and Validation

Completed on

2026-06-30

---

## Public Interfaces

REST API

GET /search

POST /download

POST /process

GET /status

Output Format

ProcessedLightCurve

Python API

- `create_mast_client()`
- `search_observations()`
- `FitsCache`
- `download_fits()`
- `read_fits()`

Structured FITS Output

- `LightCurve`
- `LightCurveMetadata`

Preprocessing API

- `remove_non_finite()`
- `filter_quality()`
- `normalize_flux()`
- `median_filter_flux()`
- `wavelet_denoise()`
- `preprocess_light_curve()`
- `PreprocessedLightCurve`

Feature and Export API

- `generate_statistics()`
- `generate_metadata()`
- `generate_feature_record()`
- `export_numpy()`
- `export_parquet()`
- `FeatureRecord`
- `DatasetMetadata`

REST Request Contracts

- `GET /search`: `target`, optional repeated `missions`, `radius_deg`, `limit`
- `POST /download`: JSON `mast_id`
- `POST /process`: JSON `fits_path`, optional `mission`, optional
  `preprocessing`
- `GET /status`: no parameters

---

## Contracts

This repository guarantees

- Stable processed output
- Stable metadata schema
- Stable REST endpoints

Breaking these interfaces requires updating both

- transitlens-ml-core
- transitlens-platform

---

## Dependencies

External

- astroquery
- astropy
- lightkurve
- pywavelets
- scipy

Internal

None

---

## Current Deliverables

Completed

- Python 3.11+ project and dependency configuration
- Ruff, Black, pytest, coverage, and pre-commit configuration
- Typed TOML and environment-based runtime settings
- Structured Loguru configuration
- FastAPI application factory (business endpoints intentionally deferred)
- Reproducible dependency lock file
- Phase 1 smoke and configuration tests
- Anonymous MAST access
- Optional MAST API-token authentication
- Injection of a caller-managed authenticated MAST client
- Typed Kepler, K2, and TESS observation search
- Deterministic observation result ordering and limits
- Collision-safe, atomic FITS download cache
- Deterministic light-curve FITS product selection
- Graceful MAST search, authentication, product, and download failures
- Phase 2 unit tests using offline Astroquery-compatible clients
- Astropy-based FITS reader with guaranteed file closure
- Mission detection for Kepler, K2, and TESS
- Deterministic light-curve table HDU selection
- TIME and mission-compatible flux extraction
- Optional quality-flag extraction
- Immutable float64 time/flux arrays and int64 quality arrays
- Structural validation for dimensions, alignment, usability, and time ordering
- Descriptive FITS read, HDU, column, mission, and validation exceptions
- Phase 3 mission fixtures and malformed-input tests
- Aligned non-finite TIME/FLUX removal
- Mission-aware Lightkurve quality filtering
- Robust positive-median flux normalization
- Conservative centered median filtering without zero-padded edges
- Adaptive PyWavelets db4 soft-threshold denoising
- Immutable `PreprocessedLightCurve` output and preprocessing provenance
- Sample accounting for non-finite and quality-filtered cadences
- Synthetic transit-depth, ingress, egress, noise, and determinism tests
- Deterministic statistical feature generation from wavelet-denoised flux
- Versioned canonical dataset metadata without runtime timestamps
- Atomic, pickle-free, byte-deterministic compressed NumPy export
- Atomic, byte-deterministic Parquet export with canonical schema metadata
- Explicit representation of absent quality flags in both formats
- NumPy and Parquet round-trip and output-consistency tests
- Injectable FastAPI service dependencies without module-level clients
- GET `/status` readiness and supported-mission response
- GET `/search` typed MAST observation search
- POST `/download` cached preferred-product retrieval
- POST `/process` FITS parsing, preprocessing, and feature generation
- Cache-contained FITS path enforcement for process requests
- Stable domain error payloads and HTTP status mappings
- REST integration tests using an in-process ASGI transport
- Search-to-export integration coverage for Kepler, K2, and TESS
- Complete REST search, download, and process workflow validation
- Whole-pipeline array, feature, NumPy, and Parquet consistency validation
- 50,000-cadence local pipeline performance benchmark
- Final repository scope and dependency audit

Verification

- Ruff passes
- Black passes
- 119 tests pass
- Test coverage: 98.05%
- Source distribution and wheel build successfully
- FastAPI application factory starts successfully
- Anonymous public MAST search and FITS download verified with Kepler-10
- Repeated public download verified to reuse the cached FITS file
- Cached public Kepler FITS parsed successfully with 476 samples
- Repeated parsing produced identical array digests
- Cached Kepler preprocessing completed in approximately 0.0013 seconds
- Cached Kepler preprocessing retained 469 of 476 cadences after removing 7
  non-finite measurements
- Repeated preprocessing produced identical array digests
- Cached Kepler feature generation produced 469-sample statistics
- Cached Kepler NumPy export produced a deterministic 14,429-byte artifact
- Cached Kepler Parquet export produced a deterministic 16,806-byte artifact
- Real cached Kepler `/process` response returned 469 processed samples and a
  matching 469-sample feature record
- `/status` and `/process` ASGI acceptance requests completed successfully
- Kepler, K2, and TESS each complete search through both export formats
- Repeated full-pipeline runs produce identical arrays, feature records, and
  artifact digests
- The 50,000-cadence parse-to-export benchmark completed in 0.12 seconds

Pending

- None

---

## Configuration Contract

Default configuration is stored in `configs/default.toml`.

Runtime values may be supplied with environment variables using the
`TRANSITLENS_` prefix. Environment variables take precedence over values from
the configuration file. MAST credentials remain optional and are never stored
in source configuration.

---

## Phase 7 Notes

- Exactly four business endpoints are exposed. Documentation and OpenAPI routes
  remain disabled so no additional endpoint paths are introduced.
- The FastAPI application uses an application factory so settings are injected
  without mutable global application state.
- Anonymous access is the default. `TRANSITLENS_MAST_API_TOKEN` enables optional
  API-token authentication without persisting the credential.
- A caller that already owns an authenticated Astroquery session can inject its
  configured client without exposing credentials to this repository.
- Cached filenames combine a stable URI digest with the MAST product filename.
  Only non-empty completed downloads are placed in the cache.
- Flux selection preference is `PDCSAP_FLUX`, then `SAP_FLUX`, then `FLUX`.
  The selected source column is always recorded in metadata.
- Kepler and K2 prefer `SAP_QUALITY`; TESS prefers `QUALITY`. Quality remains
  optional when the source product does not provide it.
- Isolated non-finite TIME or FLUX samples remain aligned in the raw light curve
  for removal by the explicitly separate Phase 4 cleaning step.
- Finite TIME samples must be strictly increasing, and every light curve must
  contain at least one finite time and flux value.
- The required operation order is enforced by `preprocess_light_curve()`:
  non-finite removal, quality filtering, normalization, median filtering, then
  wavelet denoising.
- Lightkurve's mission-specific `default` quality bitmasks are used by default;
  named `none`, `hard`, and `hardest` policies remain explicitly configurable.
- Median filtering defaults to five cadences with nearest-edge handling.
- Wavelet denoising defaults to db4, two decomposition levels, and an adaptive
  soft threshold scaled by 0.5. Only the finest detail band is thresholded to
  protect transit depth, ingress, and egress.
- Synthetic transit tests require depth preservation within 5%, ingress and
  egress RMSE below 0.0015 normalized flux, and reduced out-of-transit noise.
- Features are computed from `wavelet_flux`. Standard deviation and variance
  use population definitions (`ddof=0`). RMS is deviation from the median
  baseline. SNR is absolute mean divided by population standard deviation and
  is null for a constant signal. Cadence is the median timestamp difference.
- Dataset schema version `1.0` contains time, cleaned raw flux, normalized flux,
  median-filtered flux, wavelet flux, optional quality flags, canonical feature
  JSON, source provenance, preprocessing parameters, and sample accounting.
- NumPy artifacts use fixed ZIP member metadata and sorted members to guarantee
  byte-identical output for identical inputs.
- Parquet artifacts use explicit column order and dtypes plus canonical Arrow
  schema metadata to guarantee deterministic output under the locked runtime.
- `/process` accepts only files resolving inside the configured cache directory;
  arbitrary server filesystem reads are rejected with HTTP 403.
- `/process` returns time, cleaned raw flux, normalized flux, median-filtered
  flux, wavelet flux, optional quality flags, preprocessing metadata, and the
  canonical feature record.
- MAST authentication failures map to 401, missing FITS products to 404,
  upstream MAST failures to 502, and scientific input failures to 422.
- `/status` performs no MAST network request and therefore represents local
  application readiness only.
- The full offline integration suite exercises search, download, FITS parsing,
  validation, preprocessing, feature generation, and both exporters without
  requiring network access or credentials.
- Live anonymous MAST search and download were separately verified against
  Kepler-10 during Phase 2.
- Coverage exceeds the required 90% threshold with branch coverage enabled.
- The source and test trees contain no CNN, autoencoder, transformer, training,
  inference, frontend, database, or platform implementation.
- All seven documented phases are complete.

---

## Known Risks

- FITS formats differ between missions.
- Large downloads require caching.
- MAST authentication should remain optional for public datasets.
- Wavelet parameters must preserve transit features while reducing noise.

---

## Next Repository

transitlens-ml-core

Expected Input

ProcessedLightCurve

Fields

- time
- normalized_flux
- wavelet_flux
- metadata

The data format produced here is considered the canonical input for all machine learning models.
