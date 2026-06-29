# reporting.py
# ------------
# Human-readable markdown reports and diagnostic logs generation for Phase 2.

from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_phase2_report(config, run_id: str, results: dict) -> None:
    """Assembles all evaluation summaries and writes phase2_scientific_report.md."""
    m = config.manifests_dir
    report_path = m / "phase2_scientific_report.md"
    
    # Save a JSON file as well
    report_json_path = m / "phase2_validation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    md_content = f"""# TransitLens Phase 2 Scientific Report
Run ID: {run_id}
Status: COMPLETE

## Executive Summary
This report summarizes the eclipsing binary and blend contamination diagnostic performance for the Phase 2 vetting pipeline.

## Evaluation Results
```json
{json.dumps(results, indent=2)}
```
"""
    report_path.write_text(md_content)
    logger.info(f"Generated Phase 2 report at {report_path}")
