# TransitLens Phase 2 diagnostics

Phase 2 converts a detected periodic event into transparent eclipsing-binary and contamination evidence. It does **not** train or enable the final classifier, and its risk scores are not calibrated probabilities.

## Scientific flow

1. Verify the frozen Phase 1 manifests and checksums.
2. Select one observation per TIC using usable fraction, usable cadence count, cadence, and observation ID—never labels or classifier success.
3. Detect the ephemeris with TransitLens BLS.
4. Compute morphology, harmonics, centroid, pixel, Gaia, crowding, dilution, multi-aperture, and ephemeris-match evidence when inputs exist.
5. Preserve unavailable measurements as null plus an availability flag.
6. Tune risk thresholds on train plus validation only, freeze them, and evaluate the target-disjoint test split.

Bulk processing is offline with respect to Gaia and never downloads TPFs implicitly. Network acquisition must be explicitly scoped because TPF storage can be large.

## Commands

```powershell
python -m phase2.cli verify-phase1
python -m phase2.cli build-benchmark
python -m phase2.cli build-features --workers 8
python -m phase2.cli tune-thresholds
python -m phase2.cli evaluate
python -m phase2.cli validate
python -m phase2.cli report --run-id phase2_detected_v210
```

`--limit` writes only beneath `phase2_development/` and can never produce release evidence. Official evaluation requires `--ephemeris-mode detected`.
