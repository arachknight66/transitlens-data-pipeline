# TransitLens Phase 1 real-data pipeline

This package builds the provenance-controlled TESS high-cadence dataset. It does not train or enable a classifier.

## Scientific boundary

Only official MAST/STScI SPOC or explicitly configured TESS-SPOC light-curve FITS products can satisfy the real-observation gate. TESSCut FFI products, catalogue rows, generated FITS fixtures, synthetic curves, failed files, and duplicate alternatives are excluded from the count.

The immutable release floor is 20,000 successfully parsed TIC-sector observations. The validator cannot lower that floor through a development configuration.

## Installation

From the repository root:

```powershell
python -m pip install -r transitlens-data-pipeline/requirements.txt
$env:PYTHONPATH = "transitlens-data-pipeline"
```

## Stage commands

```powershell
python -m phase1.cli discover --config config/phase1_dataset.yaml
python -m phase1.cli select-sectors --config config/phase1_dataset.yaml
python -m phase1.cli download --config config/phase1_dataset.yaml --resume
python -m phase1.cli process --config config/phase1_dataset.yaml
python -m phase1.cli ingest-catalogs --config config/phase1_dataset.yaml
python -m phase1.cli resolve-labels --config config/phase1_dataset.yaml
python -m phase1.cli build-splits --config config/phase1_dataset.yaml
python -m phase1.cli build-manifest --config config/phase1_dataset.yaml
python -m phase1.cli validate --config config/phase1_dataset.yaml
python -m phase1.cli report --config config/phase1_dataset.yaml
```

`download` resumes `.part` files with HTTP Range requests when the server supports them and atomically promotes only readable FITS files. `process` is also resumable and reprocesses products written by an older parser contract.

Use `--limit` only for development. It never weakens the release validator. The production command has no small default limit.

## Full execution

```powershell
python -m phase1.cli run-all --config config/phase1_dataset.yaml --resume
```

Exit codes are 0 for PASS, 1 for FAIL, and 2 for PARTIAL scientific completion.

## Safe interruption and cleanup

All raw verified FITS files are preserved. It is safe to remove orphaned `.part` files only after confirming they are not needed for a resumed Range request. Quarantine copies may be removed after their source FITS checksum is confirmed in the download manifest; verified raw observations must not be deleted by cleanup scripts.

Canonical artifacts are under `data/manifests/phase1`; run logs and resolved configurations are under `runs/phase1/<run-id>`.
