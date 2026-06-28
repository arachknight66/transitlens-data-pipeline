import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

STRENGTH_SCORES = {
    "strong": 3,
    "medium": 2,
    "weak": 1,
    "none": 0
}

def resolve_labels(config):
    """
    Deterministically resolves a single canonical label for each unique TIC ID
    based on label evidence rows and policy rules.
    Outputs resolved_labels.parquet and contradictions.parquet.
    """
    config.ensure_dirs()
    manifests_dir = config.manifests_dir
    
    evidence_path = manifests_dir / "label_evidence.parquet"
    if not evidence_path.exists():
        raise FileNotFoundError(f"Label evidence file not found: {evidence_path}. Run ingestion first.")
        
    df_evidence = pd.read_parquet(evidence_path)
    
    # Load all discovered TIC IDs from discovery manifest to ensure we output a row for EVERY target
    discovery_path = manifests_dir / "discovery_manifest.parquet"
    if not discovery_path.exists():
        raise FileNotFoundError(f"Discovery manifest not found: {discovery_path}")
    df_disc = pd.read_parquet(discovery_path)
    all_tics = set(df_disc["tic_id"].unique())
    
    resolved_rows = []
    contradiction_rows = []
    
    policy_version = config.label_policy_version
    
    # Group evidence by TIC ID
    grouped = df_evidence.groupby("tic_id")
    
    for tic_id in all_tics:
        if tic_id not in grouped.groups:
            # Unlabeled target (no evidence)
            resolved_rows.append({
                "tic_id": int(tic_id),
                "resolved_label": "unlabeled",
                "label_subtype": "unlabeled",
                "winning_evidence_ids": [],
                "rejected_evidence_ids": [],
                "conflict_count": 0,
                "conflict_description": "",
                "resolution_reason": "No catalog evidence found for this target.",
                "evidence_level": "none",
                "label_strength": "none",
                "requires_review": False,
                "policy_version": policy_version
            })
            continue
            
        target_ev = grouped.get_group(tic_id)
        
        # Group evidence by candidate label
        # Exclude unlabeled/exclude categories from voting
        valid_ev = target_ev[~target_ev["canonical_label_candidate"].isin(["unlabeled", "exclude", "None"])]
        
        if len(valid_ev) == 0:
            resolved_rows.append({
                "tic_id": int(tic_id),
                "resolved_label": "unlabeled",
                "label_subtype": "unlabeled",
                "winning_evidence_ids": [],
                "rejected_evidence_ids": [],
                "conflict_count": 0,
                "conflict_description": "",
                "resolution_reason": "All available evidence maps to unlabeled or exclude.",
                "evidence_level": "none",
                "label_strength": "none",
                "requires_review": False,
                "policy_version": policy_version
            })
            continue
            
        # Compile candidate labels and their highest scores
        label_scores = {}
        for label, group in valid_ev.groupby("canonical_label_candidate"):
            max_score = 0
            for _, r in group.iterrows():
                score = STRENGTH_SCORES.get(r["evidence_strength"], 0)
                if score > max_score:
                    max_score = score
            label_scores[label] = max_score
            
        # Find the maximum score achieved across all candidate labels
        highest_score = max(label_scores.values())
        winning_labels = [lbl for lbl, scr in label_scores.items() if scr == highest_score]
        
        # Equal-strength class conflicts are never balance- or date-broken.
        # The versioned policy does not authorize a source/date precedence.
        is_contradiction = len(winning_labels) > 1
        if is_contradiction:
            winner = "review_required"
            resolution_reason = f"Equal-strength contradiction between {winning_labels}; policy requires review."
        else:
            winner = winning_labels[0]
            resolution_reason = f"Resolved via deterministic policy. Highest evidence strength is {highest_score}."

        if highest_score <= STRENGTH_SCORES["weak"]:
            winner = "review_required"
            resolution_reason = "Only weak catalogue evidence is available; supervised use requires review."
            
        # If routed to review_required, requires_review is True
        requires_review = (winner == "review_required")
        
        # Separate winning and rejected evidence
        if winner == "review_required":
            winning_ev_rows = valid_ev
            rejected_ev_rows = pd.DataFrame()
            winning_ids = winning_ev_rows["evidence_id"].tolist()
            rejected_ids = []
            max_strength = max(
                winning_ev_rows["evidence_strength"].tolist(),
                key=lambda s: STRENGTH_SCORES.get(s, 0),
            )
            max_level = "catalog_authoritative" if "catalog_authoritative" in winning_ev_rows["evidence_level"].tolist() else "catalog_weak"
            
            conflict_desc = ""
            if is_contradiction:
                conflict_desc = f"Equal strength conflict for TIC-{tic_id} between labels: {list(label_scores.keys())}"
                contradiction_rows.append({
                    "tic_id": int(tic_id),
                    "conflict_labels": list(label_scores.keys()),
                    "conflict_description": conflict_desc,
                    "evidence_ids": winning_ids,
                    "source_catalogs": winning_ev_rows["source_catalog"].tolist()
                })
        else:
            winning_ev_rows = valid_ev[valid_ev["canonical_label_candidate"] == winner]
            rejected_ev_rows = valid_ev[valid_ev["canonical_label_candidate"] != winner]
            
            winning_ids = winning_ev_rows["evidence_id"].tolist()
            rejected_ids = rejected_ev_rows["evidence_id"].tolist()
            
            # Label strength & level
            max_strength = max(winning_ev_rows["evidence_strength"].tolist(), key=lambda s: STRENGTH_SCORES.get(s, 0))
            max_level = "catalog_authoritative" if "catalog_authoritative" in winning_ev_rows["evidence_level"].tolist() else "catalog_weak"
            conflict_desc = f"Contradicting evidence from: {rejected_ev_rows['source_catalog'].tolist()}" if len(rejected_ev_rows) > 0 else ""
            
        # Determine subtype (confirmed / candidate) for exoplanet_transit
        subtype = "unlabeled"
        if winner == "exoplanet_transit":
            # If any winning evidence is CONFIRMED or strong, label_subtype is confirmed, else candidate
            orig_disps = [str(d).upper() for d in winning_ev_rows["original_disposition"].tolist()]
            if any(d in ("CP", "KP", "CONFIRMED") for d in orig_disps):
                subtype = "confirmed"
            else:
                subtype = "candidate"
        elif winner == "eclipsing_binary":
            subtype = "eclipsing_binary"
        elif winner == "blend_contamination":
            subtype = "blend_contamination"
        elif winner == "stellar_variability_or_other":
            subtype = "stellar_variability_or_other"
        elif winner == "review_required":
            subtype = "review_required"

        resolved_rows.append({
            "tic_id": int(tic_id),
            "resolved_label": winner,
            "label_subtype": subtype,
            "winning_evidence_ids": winning_ids,
            "rejected_evidence_ids": rejected_ids,
            "conflict_count": len(rejected_ids),
            "conflict_description": conflict_desc,
            "resolution_reason": resolution_reason,
            "evidence_level": max_level,
            "label_strength": max_strength,
            "requires_review": requires_review,
            "policy_version": policy_version,
            "source_catalogs": sorted(set(winning_ev_rows["source_catalog"].astype(str))),
            "source_catalog_versions": sorted(set(winning_ev_rows["source_version"].astype(str))),
            "source_record_identifiers": sorted(set(winning_ev_rows["source_row_identifier"].astype(str))),
            "dispositions": sorted(set(winning_ev_rows["original_disposition"].astype(str))),
            "disposition_dates": sorted(set(winning_ev_rows["disposition_date"].astype(str))),
            "catalogue_checksums": sorted(set(winning_ev_rows["source_checksum"].astype(str))),
        })
        
    df_resolved = pd.DataFrame(resolved_rows)
    output_path = manifests_dir / "resolved_labels.parquet"
    df_resolved.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(df_resolved)} resolved label rows to {output_path}")
    
    # Save contradictions manifest
    df_contr = pd.DataFrame(contradiction_rows)
    if len(df_contr) == 0:
        df_contr = pd.DataFrame(columns=["tic_id", "conflict_labels", "conflict_description", "evidence_ids", "source_catalogs"])
    contr_path = manifests_dir / "contradictions.parquet"
    df_contr.to_parquet(contr_path, index=False)
    logger.info(f"Wrote {len(df_contr)} contradiction log rows to {contr_path}")
    
    return df_resolved
