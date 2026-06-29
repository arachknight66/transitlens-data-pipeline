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
    parser.add_argument("--run-id",default="phase2_run")
    args=parser.parse_args(argv); config=Phase2Config(args.config); config.ensure_dirs(); m=config.manifests_dir
    thresholds=config.REPO_ROOT/"transitlens-ml-core"/"config"/"threshold_registry.yaml"
    try:
        if args.command=="verify-phase1": result=verify_phase1(config)
        elif args.command in {"build-benchmark","select-tpf-benchmark"}: result=build_benchmark_manifest(config)
        elif args.command in {"build-features","compute-all","compute-morphology","compute-spatial"}:
            result=materialize_features(config,args.limit,workers=args.workers,split=args.split,resume=args.resume,
                                        ephemeris_mode=args.ephemeris_mode,dry_run=args.dry_run)
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
            features=materialize_features(config,args.limit,workers=args.workers,ephemeris_mode=args.ephemeris_mode,dry_run=args.dry_run)
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
        elif args.command in {"discover-tpfs","download-tpfs","verify-tpfs","query-gaia"}:
            raise ValueError(f"{args.command} requires explicit acquisition scope and is not run implicitly; use --tic-id and the dedicated cached product APIs")
        else: raise ValueError(f"unsupported command: {args.command}")
        print(json.dumps(result,indent=2,default=str)); return 0 if result.get("status") in {"PASS","COMPLETE","SUCCESS"} else 2
    except Exception as exc:
        logger.exception("Phase 2 command failed")
        print(json.dumps({"status":"FAIL","error":f"{type(exc).__name__}: {exc}"},indent=2)); return 1

if __name__=="__main__": raise SystemExit(main())
