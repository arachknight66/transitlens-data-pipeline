import os
import sys
import argparse
import json
import platform
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from phase1.atomic_io import atomic_write_parquet

# Add repo root to sys.path to resolve imports properly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase1.config import Config
import phase1.archive_discovery as discovery
import phase1.downloader as downloader
import phase1.fits_parser as parser
import phase1.catalog_ingestion as ingestion
import phase1.label_resolver as resolver
import phase1.split_builder as splits
import phase1.manifest as manifest_builder
import phase1.validation as validator
import phase1.reporting as reporting
import phase1.checksums as csums

# Set up logging to stdout and JSON files
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("phase1")

def get_run_id():
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

def setup_run_logging(config, run_id):
    """Sets up log files under runs/phase1/<run_id>/."""
    run_dir = config.REPO_ROOT / "runs" / "phase1" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = run_dir / "execution.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    
    # Add handler to root logger or our specific logger
    logger.addHandler(file_handler)
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "run_id": run_id,
            }, sort_keys=True)
    json_handler = logging.FileHandler(run_dir / "execution.jsonl", encoding="utf-8")
    json_handler.setFormatter(JsonFormatter())
    logging.getLogger().addHandler(json_handler)
    shutil.copy2(config.config_path, run_dir / "resolved_config.yaml")
    logger.info(f"Initialized run directory and logging at: {run_dir}")
    return run_dir

def run_process_stage(config, limit=None, run_id=None):
    """Iterates through verified files in download manifest and processes them into NPZs concurrently."""
    config.ensure_dirs()
    manifests_dir = config.manifests_dir
    download_manifest_path = manifests_dir / "download_manifest.parquet"
    
    if not download_manifest_path.exists():
        raise FileNotFoundError(f"Download manifest not found: {download_manifest_path}. Run downloader first.")
        
    df_dl = pd.read_parquet(download_manifest_path)

    # Migrate the prototype's overloaded terminal state into independent
    # download/parse states. Download validity must survive parsing.
    if "download_status" not in df_dl.columns:
        df_dl["download_status"] = df_dl["final_status"].replace({"processed": "verified"})
    if "parse_status" not in df_dl.columns:
        df_dl["parse_status"] = df_dl["final_status"].map({"processed": "success"}).fillna("pending")

    def needs_processing(row):
        if row["download_status"] != "verified":
            return False
        tic_id, sector = int(row["tic_id"]), int(row["sector"])
        sidecar = config.processed_dir / "metadata" / f"TIC-{tic_id:012d}_sector-{sector:04d}_lc_meta.json"
        if not sidecar.exists():
            return True
        try:
            with open(sidecar, "r", encoding="utf-8") as handle:
                return json.load(handle).get("parser_version") != "1.1.0"
        except Exception:
            return True

    targets = df_dl[df_dl.apply(needs_processing, axis=1)].copy()
    if limit is not None:
        targets = targets.head(int(limit))
        
    if len(targets) == 0:
        logger.info("No verified targets found for processing.")
        return
        
    logger.info(f"Starting concurrent FITS processing for {len(targets)} targets...")
    
    processed_count = 0
    quarantine_count = 0
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Use 8 workers to balance CPU utilization and Disk I/O
    concurrency = min(8, os.cpu_count() or 4)
    logger.info(f"Using ThreadPoolExecutor with {concurrency} workers.")
    
    t_start = time.time()
    
    def worker(item):
        idx, row = item
        res = parser.process_and_save(idx, row, config, manifests_dir, download_manifest_path)
        return idx, res
        
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(worker, item): item for item in targets.iterrows()}
        
        for future in as_completed(futures):
            idx, res = future.result()
            
            df_dl.at[idx, "parse_status"] = res["parse_status"]
            df_dl.at[idx, "final_status"] = "verified" if res["status"] == "processed" else "quarantined"
            df_dl.at[idx, "failure_message"] = res["error_msg"]
            
            if res["status"] == "processed":
                processed_count += 1
            else:
                quarantine_count += 1
                
            total = processed_count + quarantine_count
            if total % 1000 == 0 or total == len(targets):
                logger.info(f"Processed: {processed_count}, Quarantined: {quarantine_count} of {len(targets)} files. Elapsed: {time.time() - t_start:.1f}s")
                # Update download manifest parquet incrementally
                atomic_write_parquet(df_dl, download_manifest_path, index=False)
                
    atomic_write_parquet(df_dl, download_manifest_path, index=False)
    logger.info(f"FITS parsing stage completed: {processed_count} successfully parsed, {quarantine_count} quarantined.")

def main():
    parser_arg = argparse.ArgumentParser(description="TransitLens Phase 1 Dataset Pipeline CLI")
    parser_arg.add_argument("command", choices=[
        "discover", "select-sectors", "ingest-catalogs", "resolve-labels", "download", 
        "verify-downloads", "process", "build-splits", "build-manifest", 
        "validate", "report", "final-verify", "status", "run-all",
        "discover-dvr-xml", "download-dvr-xml", "discover-supplement"
    ], help="Stage command to execute")
    parser_arg.add_argument("--config", default=None, help="Path to config YAML")
    parser_arg.add_argument("--run-id", default=None, help="Run ID for tracking execution")
    parser_arg.add_argument("--limit", type=int, default=None, help="Limit number of processed lightcurves (useful for mock/dev runs)")
    parser_arg.add_argument("--sector", type=int, default=None, help="Restrict to a specific sector")
    parser_arg.add_argument("--resume", action="store_true", default=True, help="Resume download/processing from manifest")
    parser_arg.add_argument("--retry-failures", action="store_true", default=False, help="Retry failed downloads")
    parser_arg.add_argument("--concurrency", type=int, default=None, help="Download concurrency")
    parser_arg.add_argument("--dry-run", action="store_true", default=False, help="Dry run mode")
    parser_arg.add_argument("--verify-only", action="store_true", default=False, help="Verify files without downloading")
    parser_arg.add_argument("--supplement-only", action="store_true", default=False, help="Restrict downloading to the frozen labelled supplementary cohort")
    
    args = parser_arg.parse_args()
    
    config = Config(args.config)
    
    if args.concurrency is not None:
        config.download_concurrency = args.concurrency
        
    run_id = args.run_id if args.run_id is not None else get_run_id()
    run_dir = setup_run_logging(config, run_id)
    
    # Save the execution run summary info
    run_info = {
        "run_id": run_id,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config.config_path),
        "random_seed": config.random_seed,
        "sectors": config.selected_sectors,
        "limits": args.limit,
        "stage": args.command,
        "arguments": vars(args),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "dataset_version": config.dataset_version,
        "label_policy_version": config.label_policy_version,
    }
    
    logger.info(f"Running pipeline stage: '{args.command}' with Run ID: '{run_id}'")
    
    try:
        if args.command == "discover":
            discovery.discover_candidates(config)

        elif args.command == "select-sectors":
            from phase1.sector_selection import select_sectors
            selected = select_sectors(config)
            logger.info(f"Selected sectors: {selected['sector'].astype(int).tolist()}")
            
        elif args.command == "ingest-catalogs":
            ingestion.ingest_all_catalogs(config)
            
        elif args.command == "resolve-labels":
            resolver.resolve_labels(config)
            
        elif args.command == "download":
            downloader.run_download(
                config, limit=args.limit, sector=args.sector, 
                resume=args.resume, retry_failures=args.retry_failures,
                dry_run=args.dry_run, supplement_only=args.supplement_only
            )
            
        elif args.command == "verify-downloads":
            downloader.run_download(
                config, limit=args.limit, sector=args.sector, 
                resume=args.resume, retry_failures=args.retry_failures,
                dry_run=args.dry_run, verify_only=True
            )
            
        elif args.command == "process":
            run_process_stage(config, limit=args.limit, run_id=run_id)
            
        elif args.command == "build-splits":
            splits.build_splits(config)
            
        elif args.command == "build-manifest":
            # Determine which TICs are selected representatives
            download_manifest_path = config.manifests_dir / "download_manifest.parquet"
            if download_manifest_path.exists():
                df_dl = pd.read_parquet(download_manifest_path)
                valid_obs = df_dl[df_dl["final_status"] == "verified"]
                
                # Deduplicate to find selected observation IDs
                import phase1.deduplication as dedup
                selected_obs_ids = dedup.resolve_duplicates(config)
            else:
                selected_obs_ids = set()
                
            manifest_builder.build_observation_manifest(config, selected_obs_ids, run_id)
            
        elif args.command == "validate":
            res = validator.run_release_validation(config)
            logger.info(f"Release validation status: {res['status']}")
            if res["status"] == "FAIL":
                sys.exit(1)
            if res["status"] == "PARTIAL":
                sys.exit(2)
                
        elif args.command == "report":
            reporting.generate_release_documentation(config, run_id)

        elif args.command == "final-verify":
            from phase1.final_verification import run_final_verification
            result = run_final_verification(config)
            logger.info(f"Independent final verification status: {result['status']}")
            if result["status"] == "FAIL":
                sys.exit(1)
            if result["status"] == "PARTIAL":
                sys.exit(2)

        elif args.command == "discover-dvr-xml":
            from phase1.dv_xml import discover_targeted_dvr_xml
            manifest, missing = discover_targeted_dvr_xml(config)
            logger.info(f"Discovered {len(manifest)} targeted DVR XML products; missing pairs: {missing}")

        elif args.command == "download-dvr-xml":
            from phase1.dv_xml import download_targeted_dvr_xml
            manifest = download_targeted_dvr_xml(config, concurrency=args.concurrency or 4)
            logger.info(f"DVR XML statuses: {manifest['status'].value_counts().to_dict()}")

        elif args.command == "discover-supplement":
            from phase1.supplementary_cohort import discover_and_merge_supplement
            selected, summary = discover_and_merge_supplement(config)
            logger.info(f"Selected {len(selected)} supplementary products: {summary}")
            
        elif args.command == "status":
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Config path: {config.config_path}")
            logger.info(f"Raw storage path: {config.raw_dir}")
            logger.info(f"Processed storage path: {config.processed_dir}")
            for filename in ("discovery_manifest.parquet", "download_manifest.parquet", "observation_manifest.parquet"):
                path = config.manifests_dir / filename
                if path.exists():
                    frame = pd.read_parquet(path)
                    logger.info(f"{filename}: {len(frame)} rows")
                    for column in ("final_status", "download_status", "parse_status"):
                        if column in frame.columns:
                            logger.info(f"  {column}: {frame[column].value_counts(dropna=False).to_dict()}")
            
        elif args.command == "run-all":
            logger.info("Executing entire Phase 1 pipeline sequentially...")
            
            logger.info("--- [Stage 1/9] Archive Discovery ---")
            discovery.discover_candidates(config)

            from phase1.sector_selection import select_sectors
            select_sectors(config)
            
            logger.info("--- [Stage 2/9] Catalog Ingestion ---")
            ingestion.ingest_all_catalogs(config)
            
            logger.info("--- [Stage 3/9] Label Resolution ---")
            resolver.resolve_labels(config)
            
            logger.info("--- [Stage 4/9] Concurrent Downloader ---")
            existing_verified = 0
            existing_download_manifest = config.manifests_dir / "download_manifest.parquet"
            if existing_download_manifest.exists():
                existing_downloads = pd.read_parquet(existing_download_manifest)
                status_column = "download_status" if "download_status" in existing_downloads else "final_status"
                existing_verified = int((existing_downloads[status_column].isin(["verified", "processed"])).sum())
            if existing_verified >= config.minimum_successful_observations and args.limit is None:
                logger.info(
                    f"Skipping additional acquisition: {existing_verified} verified products already meet "
                    f"the {config.minimum_successful_observations} observation target."
                )
            else:
                downloader.run_download(
                    config, limit=args.limit, sector=args.sector,
                    resume=args.resume, retry_failures=args.retry_failures,
                    dry_run=args.dry_run
                )
            
            logger.info("--- [Stage 5/9] FITS Parsing & Normalization ---")
            run_process_stage(config, limit=args.limit, run_id=run_id)
            
            logger.info("--- [Stage 6/9] Duplicate Resolution ---")
            import phase1.deduplication as dedup
            selected_obs_ids = dedup.resolve_duplicates(config)
            
            logger.info("--- [Stage 7/9] Split Generation ---")
            splits.build_splits(config)
            
            logger.info("--- [Stage 8/9] Canonical Manifest Compilation ---")
            manifest_builder.build_observation_manifest(config, selected_obs_ids, run_id)
            
            logger.info("--- [Stage 9/9] Cryptographic Verification & release validation ---")
            csums.generate_checksums_file(config)
            res = validator.run_release_validation(config)
            
            logger.info("--- Generation of documentation reports ---")
            reporting.generate_release_documentation(config, run_id)

            from phase1.final_verification import run_final_verification
            final_result = run_final_verification(config)
            if final_result["status"] == "FAIL":
                res["status"] = "FAIL"
            
            # Save end time
            run_info["end_time"] = datetime.now(timezone.utc).isoformat()
            run_info["status"] = res["status"]
            with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
                json.dump(run_info, f, indent=2)
                
            logger.info(f"Phase 1 Pipeline sequence finished. Release Validation Status: {res['status']}")
            
            if res["status"] == "FAIL":
                logger.error("Pipeline finished with release-blocking failures.")
                sys.exit(1)
            if res["status"] == "PARTIAL":
                logger.warning("Pipeline completed with scientific shortfalls; release remains PARTIAL.")
                sys.exit(2)
                
    except Exception as e:
        logger.error(f"Stage '{args.command}' failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
