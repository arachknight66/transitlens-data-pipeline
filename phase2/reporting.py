"""Single-source Phase 2 release reporting; never hardcodes completion."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import numpy as np

def _json_default(value):
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")

def generate_phase2_report(config, run_id: str, results: dict) -> None:
    m=config.manifests_dir
    record={"run_id":run_id,"generated_at":datetime.now(timezone.utc).isoformat(),**results}
    (m/"phase2_validation_report.json").write_text(json.dumps(record,indent=2,default=_json_default),encoding="utf-8")
    metrics=record.get("evaluation",{})
    blockers=record.get("errors",[])+record.get("warnings",[])+record.get("gate_failures",[])
    lines=["# TransitLens Phase 2 scientific report","",f"**Status: {record.get('status','PARTIAL')}**","",
           "Phase 2 diagnostics are risk evidence, not calibrated astrophysical probabilities and not planet confirmation.","",
           "## Frozen evaluation record","","```json",json.dumps(metrics,indent=2,default=_json_default),"```","","## Release blockers","",]
    lines += [f"- {item}" for item in blockers] or ["- None"]
    lines += ["","## Scientific safeguards","",
              "Official features use TransitLens-detected ephemerides, preserve target-disjoint Phase 1 splits, keep labels in a separate metadata table, and represent unavailable measurements as null plus availability flags.",""]
    (m/"phase2_scientific_report.md").write_text("\n".join(lines),encoding="utf-8")
