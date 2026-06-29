from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from phase2.feature_materializer import FEATURE_COLUMNS, BOOLEAN_FEATURES, _best_observation_per_target
from phase2.evaluation import binary_metrics, tune_thresholds, evaluate_blind, promotion_gates

def test_feature_contract_has_explicit_availability():
    required={"odd_even_available","secondary_available","morphology_available","harmonic_available",
              "centroid_available","difference_image_available","gaia_available","crowding_available","multi_aperture_available"}
    assert required <= set(FEATURE_COLUMNS)

def test_truth_and_identifiers_are_not_model_features():
    prohibited={"canonical_label","resolved_label","disposition","source_catalogs","tic_id","observation_id","split"}
    assert not (prohibited & set(FEATURE_COLUMNS))

def test_best_observation_selection_is_label_independent():
    frame=pd.DataFrame({"tic_id":[1,1,2],"observation_id":["bad","good","only"],"usable_fraction":[.5,.9,.8],
                        "n_points_usable":[100,90,80],"median_cadence_seconds":[120,120,120],
                        "resolved_label":["eclipsing_binary","eclipsing_binary","exoplanet_transit"]})
    chosen=_best_observation_per_target(frame)
    assert chosen.loc[chosen.tic_id.eq(1),"observation_id"].item()=="good"

def test_binary_metrics_reports_denominators_and_counts():
    frame=pd.DataFrame({"canonical_label":["eclipsing_binary","eclipsing_binary","exoplanet_transit"],"score":[.9,.1,.8]})
    result=binary_metrics(frame,"eclipsing_binary","score",.5)
    assert (result["tp"],result["fp"],result["fn"],result["positive_support"])==(1,1,1,2)
    assert result["recall"]==pytest.approx(.5)

def _write_eval_fixture(root: Path):
    meta=[]
    for split in ["train","val","test"]:
        rows=[]
        for i,label in enumerate(["exoplanet_transit","eclipsing_binary","blend_contamination","stellar_variability_or_other"]):
            oid=f"{split}-{i}"; rows.append({"tic_id":1000+len(meta),"observation_id":oid,"eb_risk_score":.9 if label=="eclipsing_binary" else .1,
                                             "blend_risk_score":.9 if label=="blend_contamination" else .1,"review_required":False,
                                             "centroid_available":True})
            meta.append({"observation_id":oid,"canonical_label":label})
        name={"train":"phase2_features_train.parquet","val":"phase2_features_validation.parquet","test":"phase2_features_test.parquet"}[split]
        pd.DataFrame(rows).to_parquet(root/name,index=False)
    pd.DataFrame(meta).to_parquet(root/"phase2_feature_metadata.parquet",index=False)

def test_thresholds_are_selected_without_test_labels(tmp_path):
    _write_eval_fixture(tmp_path); registry=tmp_path/"thresholds.yaml"
    policy=tune_thresholds(tmp_path,registry)
    assert policy["selection_splits"]==["train","validation"] and not policy["blind_test_used"]

def test_blind_evaluation_uses_frozen_policy(tmp_path):
    _write_eval_fixture(tmp_path); registry=tmp_path/"thresholds.yaml"; tune_thresholds(tmp_path,registry)
    result=evaluate_blind(tmp_path,registry)
    assert result["targets"]==4 and result["eb"]["tp"]==1 and result["blend"]["tp"]==1

def test_support_shortage_blocks_pass():
    result={"eb":{"recall":1,"tp":1,"positive_support":1},"blend":{"recall":1,"precision":1,"tp":1,"fp":0,"positive_support":1},
            "clean_planet_false_blend_rate":0,"clean_planet_support":1}
    status,failures=promotion_gates(result,"PASS")
    assert status=="PARTIAL" and any("blind support" in item for item in failures)

def test_development_boolean_contract_is_complete():
    assert all(name in FEATURE_COLUMNS or name=="candidate_detected" for name in BOOLEAN_FEATURES)
