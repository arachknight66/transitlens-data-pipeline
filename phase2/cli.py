"""Coherent, fail-closed Phase 2 command-line interface."""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import pandas as pd

from phase2.config import Phase2Config
from phase2.benchmark_builder import build_benchmark_manifest
from phase2.feature_materializer import materialize_features
from phase2.validation import run_phase2_validation
from phase2.reporting import generate_phase2_report
from phase2.evaluation import tune_thresholds, evaluate_blind, promotion_gates
from phase2.acquisition import build_acquisition_manifest, estimate_manifest_sizes, download_manifest, spatial_critical_subset

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger=logging.getLogger("phase2")
COMMANDS=["verify-phase1","select-tpf-benchmark","discover-tpfs","download-tpfs","verify-tpfs","query-gaia",
          "compute-morphology","compute-spatial","compute-all","build-benchmark","tune-thresholds",
          "freeze-thresholds","evaluate","build-features","validate","report","status","run-all"]

def verify_phase1(config) -> dict:
    m=config.manifests_dir
    validation=json.loads((m/"validation_report.json").read_text())
    integrity=json.loads((m/"split_integrity_report.json").read_text())
    observations=pd.read_parquet(m/"observation_manifest.parquet")
    labels=pd.read_parquet(m/"resolved_labels.parquet")
    parsed=observations[observations.parse_status.eq("success")]
    provenance=labels[labels.resolved_label.isin(["exoplanet_transit","eclipsing_binary","blend_contamination","stellar_variability_or_other"])]
    checks={"parsed_observations_at_least_20000":bool(len(parsed)>=20000),
            "raw_checksums_present":bool(parsed.raw_sha256.fillna("").ne("").all()),
            "processed_checksums_present":bool(parsed.processed_sha256.fillna("").ne("").all()),
            "target_disjoint":not integrity.get("leakage_detected",True),
            "authoritative_provenance":bool(provenance.evidence_level.eq("catalog_authoritative").all()),
            "phase1_release_pass":validation.get("status")=="PASS"}
    return {"status":"PASS" if all(checks.values()) else "PARTIAL","checks":checks,"parsed_observations":len(parsed),
            "unique_tics":int(parsed.tic_id.nunique()),"phase1_reported_status":validation.get("status")}

def main(argv=None):
    parser=argparse.ArgumentParser(description="TransitLens Phase 2 scientific diagnostics")
    parser.add_argument("command",choices=COMMANDS); parser.add_argument("--config")
    parser.add_argument("--split",choices=["train","validation","test"]); parser.add_argument("--sector",type=int)
    parser.add_argument("--tic-id",type=int); parser.add_argument("--limit",type=int); parser.add_argument("--resume",action="store_true")
    parser.add_argument("--retry-failures",action="store_true"); parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--ephemeris-mode",default="detected",
                        choices=["detected","catalog_debug","injected_truth"])
    parser.add_argument("--evidence-tier",default="real_held_out"); parser.add_argument("--output-dir")
    parser.add_argument("--acquisition-scope",choices=["full","spatial-critical"],default="full")
    parser.add_argument("--run-id",default="phase2_run")
    args=parser.parse_args(argv); config=Phase2Config(args.config); config.ensure_dirs(); m=config.manifests_dir
    thresholds=config.REPO_ROOT/"transitlens-ml-core"/"config"/"threshold_registry.yaml"
    try:
        if args.command=="verify-phase1": result=verify_phase1(config)
        elif args.command in {"build-benchmark","select-tpf-benchmark"}: result=build_benchmark_manifest(config)
        elif args.command in {"build-features","compute-all","compute-morphology","compute-spatial"}:
            result=materialize_features(config,args.limit,workers=args.workers,split=args.split,resume=args.resume,
                                        ephemeris_mode=args.ephemeris_mode,dry_run=args.dry_run,output_dir=args.output_dir)
        elif args.command in {"tune-thresholds","freeze-thresholds"}:
            if args.limit: raise ValueError("development-limited runs cannot freeze thresholds")
            result=tune_thresholds(m,thresholds)
        elif args.command=="evaluate":
            if args.limit: raise ValueError("development-limited runs cannot inspect the blind test")
            result=evaluate_blind(m,thresholds)
        elif args.command=="validate": result=run_phase2_validation(config,development_only=bool(args.limit))
        elif args.command=="report":
            validation=run_phase2_validation(config,development_only=bool(args.limit)); phase1=verify_phase1(config)
            evaluation=evaluate_blind(m,thresholds) if thresholds.exists() and "frozen: true" in thresholds.read_text().lower() else {}
            status,failures=promotion_gates(evaluation,phase1["phase1_reported_status"]) if evaluation else ("PARTIAL",["blind evaluation unavailable"])
            result={**validation,"status":status if validation["status"]!="FAIL" else "FAIL","evaluation":evaluation,"gate_failures":failures}
            generate_phase2_report(config,args.run_id,result)
        elif args.command=="status":
            report=m/"phase2_validation_report.json"; result=json.loads(report.read_text()) if report.exists() else {"status":"NOT_RUN"}
        elif args.command=="run-all":
            phase1=verify_phase1(config); benchmark=build_benchmark_manifest(config)
            features=materialize_features(config,args.limit,workers=args.workers,ephemeris_mode=args.ephemeris_mode,
                                          dry_run=args.dry_run,output_dir=args.output_dir)
            if args.output_dir and not args.dry_run:
                prototype=Path(args.output_dir).resolve(); policy=tune_thresholds(prototype,prototype/"threshold_registry.yaml",("val",))
                evaluation=evaluate_blind(prototype,prototype/"threshold_registry.yaml")
                result={"status":"DEVELOPMENT_ONLY","phase1":phase1,"benchmark":benchmark,"features":features,
                        "threshold_policy":policy,"evaluation":evaluation,
                        "limitations":["partial TPF coverage","Gaia cache unavailable","not a frozen scientific release"]}
                (prototype/"prototype_report.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
                print(json.dumps(result,indent=2,default=str)); return 2
            if args.dry_run: result={"status":"DEVELOPMENT_ONLY","phase1":phase1,"benchmark":benchmark,"features":features}
            elif args.limit:
                validation=run_phase2_validation(config,development_only=True)
                result={**validation,"status":"PARTIAL","phase1":phase1,"benchmark":benchmark,"features":features,
                        "gate_failures":["development limit used; thresholds and blind test not evaluated"]}
                generate_phase2_report(config,args.run_id,result)
            else:
                policy=tune_thresholds(m,thresholds); evaluation=evaluate_blind(m,thresholds)
                validation=run_phase2_validation(config); status,failures=promotion_gates(evaluation,phase1["phase1_reported_status"])
                result={**validation,"status":status if validation["status"]!="FAIL" else "FAIL","phase1":phase1,
                        "benchmark":benchmark,"features":features,"threshold_policy":policy,"evaluation":evaluation,"gate_failures":failures}
                generate_phase2_report(config,args.run_id,result)
        elif args.command=="discover-tpfs":
            splits=("val","test") if args.split is None else (("val" if args.split=="validation" else args.split),)
            manifest=estimate_manifest_sizes(build_acquisition_manifest(config,splits),workers=args.workers)
            manifest.to_parquet(m/"tpf_discovery_manifest.parquet",index=False)
            known=manifest.expected_bytes.dropna()
            result={"status":"SUCCESS","targets":len(manifest),"sizes_resolved":len(known),
                    "expected_gb":float(known.sum()/1e9),"size_query_failures":int(manifest.expected_bytes.isna().sum()),
                    "free_gb":float(__import__('shutil').disk_usage(config.tpf_dir).free/1e9)}
        elif args.command=="download-tpfs":
            path=m/"tpf_discovery_manifest.parquet"
            if not path.exists(): raise ValueError("run discover-tpfs first")
            manifest=pd.read_parquet(path)
            if args.acquisition_scope=="spatial-critical": manifest=spatial_critical_subset(config,manifest)
            result={"status":"SUCCESS",**download_manifest(config,manifest,workers=args.workers,
                    progress_path=m/"tpf_download_progress.json")}
        elif args.command=="verify-tpfs":
            path=m/"tpf_acquisition_manifest.parquet"
            if not path.exists(): raise ValueError("no acquisition manifest")
            acquired=pd.read_parquet(path); exists=acquired.local_path.map(lambda p:Path(p).exists())
            result={"status":"PASS" if exists.all() else "PARTIAL","verified_files":int(exists.sum()),"missing_files":int((~exists).sum())}
        elif args.command=="query-gaia":
            raise ValueError("bulk Gaia acquisition is not yet started; TPF spatial evidence is being acquired first")
        else: raise ValueError(f"unsupported command: {args.command}")
        print(json.dumps(result,indent=2,default=str)); return 0 if result.get("status") in {"PASS","COMPLETE","SUCCESS"} else 2
    except Exception as exc:
        logger.exception("Phase 2 command failed")
        print(json.dumps({"status":"FAIL","error":f"{type(exc).__name__}: {exc}"},indent=2)); return 1

if __name__=="__main__": raise SystemExit(main())
