"""Target-level threshold selection and one-shot held-out evaluation."""
from __future__ import annotations
from math import sqrt
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import yaml

def _joined(manifests: Path, split: str) -> pd.DataFrame:
    filename={"train":"phase2_features_train.parquet","val":"phase2_features_validation.parquet","test":"phase2_features_test.parquet"}[split]
    features=pd.read_parquet(manifests/filename)
    metadata=pd.read_parquet(manifests/"phase2_feature_metadata.parquet")
    return features.merge(metadata[["observation_id","canonical_label"]],on="observation_id",how="left",validate="one_to_one")

def _wilson(successes: int, total: int, z: float=1.96) -> list[float|None]:
    if total == 0: return [None,None]
    p=successes/total; den=1+z*z/total
    centre=(p+z*z/(2*total))/den; radius=z*sqrt((p*(1-p)+z*z/(4*total))/total)/den
    return [max(0.,centre-radius),min(1.,centre+radius)]

def binary_metrics(frame: pd.DataFrame, positive: str, score: str, threshold: float) -> dict:
    y=frame.canonical_label.eq(positive).to_numpy(); values=frame[score].fillna(-np.inf).to_numpy(); pred=values>=threshold
    tp=int(np.sum(y&pred)); fp=int(np.sum(~y&pred)); fn=int(np.sum(y&~pred)); tn=int(np.sum(~y&~pred))
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {"threshold":threshold,"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":precision,"recall":recall,"f1":f1,
            "positive_support":int(y.sum()),"negative_support":int((~y).sum()),
            "precision_95ci":_wilson(tp,tp+fp),"recall_95ci":_wilson(tp,tp+fn)}

def tune_thresholds(manifests: Path, output: Path, selection_splits=("train","val")) -> dict:
    development=pd.concat([_joined(manifests,split) for split in selection_splits],ignore_index=True)
    grid=np.linspace(0.05,0.95,91)
    def choose(label,score,minimum_precision=0.):
        candidates=[binary_metrics(development,label,score,float(t)) for t in grid]
        eligible=[x for x in candidates if x["precision"]>=minimum_precision]
        return max(eligible or candidates,key=lambda x:(x["f1"],x["recall"],x["precision"]))
    eb=choose("eclipsing_binary","eb_risk_score")
    blend=choose("blend_contamination","blend_risk_score",0.75)
    display_splits=["validation" if split=="val" else split for split in selection_splits]
    registry={"threshold_policy_version":"2.2.0","frozen":True,"selection_splits":display_splits,
              "blind_test_used":False,"ephemeris_mode":"detected","eb_risk_threshold":eb["threshold"],
              "blend_risk_threshold":blend["threshold"],"development_metrics":{"eb":eb,"blend":blend}}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(yaml.safe_dump(registry,sort_keys=False),encoding="utf-8")
    (manifests/"threshold_registry.yaml").write_text(yaml.safe_dump(registry,sort_keys=False),encoding="utf-8")
    return registry

def evaluate_blind(manifests: Path, threshold_registry: Path) -> dict:
    policy=yaml.safe_load(threshold_registry.read_text())
    if not policy.get("frozen") or policy.get("blind_test_used"): raise ValueError("threshold registry is not a valid pre-test freeze")
    input_hashes={name:hashlib.sha256((manifests/name).read_bytes()).hexdigest() for name in
                  ["phase2_features_test.parquet","phase2_feature_metadata.parquet"]}
    input_hashes["threshold_registry.yaml"]=hashlib.sha256(threshold_registry.read_bytes()).hexdigest()
    frozen_record=manifests/"phase2_blind_evaluation.json"
    if frozen_record.exists():
        prior=json.loads(frozen_record.read_text())
        if prior.get("input_hashes")==input_hashes: return prior
    test=_joined(manifests,"test")
    eb=binary_metrics(test,"eclipsing_binary","eb_risk_score",float(policy["eb_risk_threshold"]))
    blend=binary_metrics(test,"blend_contamination","blend_risk_score",float(policy["blend_risk_threshold"]))
    planets=test.canonical_label.eq("exoplanet_transit"); blend_flag=test.blend_risk_score.fillna(-np.inf)>=policy["blend_risk_threshold"]
    false_planets=int((planets&blend_flag).sum()); planet_count=int(planets.sum())
    availability={name:int(test[name].fillna(False).sum()) for name in test.columns if name.endswith("_available")}
    sector_support=test["sector"].value_counts().sort_index().to_dict() if "sector" in test else {}
    result={"evidence_tier":"real_held_out","ephemeris_mode":"detected","targets":len(test),"input_hashes":input_hashes,
            "class_support":test.canonical_label.value_counts().to_dict(),"sector_support":sector_support,"eb":eb,"blend":blend,
            "clean_planet_false_blend_flags":false_planets,"clean_planet_support":planet_count,
            "clean_planet_false_blend_rate":false_planets/planet_count if planet_count else None,
            "review_required":int(test.review_required.fillna(True).sum()),"availability":availability}
    test.assign(eb_flag=test.eb_risk_score.fillna(-np.inf)>=policy["eb_risk_threshold"],
                blend_flag=blend_flag).to_parquet(manifests/"phase2_blind_predictions.parquet",index=False)
    missing=[]
    for column,count in availability.items():
        missing.append({"diagnostic":column.removesuffix("_available"),"available":count,"unavailable":len(test)-count,
                        "denominator":len(test),"reason":"not available from source product or insufficient detected-event data"})
    pd.DataFrame(missing).to_csv(manifests/"phase2_missingness.csv",index=False)
    frozen_record.write_text(json.dumps(result,indent=2),encoding="utf-8")
    _write_plots(test,policy,manifests)
    return result

def _write_plots(test: pd.DataFrame, policy: dict, output: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    plot_dir=output/"phase2_plots"; plot_dir.mkdir(exist_ok=True)
    for label,score,title in [("eclipsing_binary","eb_risk_score","EB risk score"),("blend_contamination","blend_risk_score","Blend risk score")]:
        y=test.canonical_label.eq(label).astype(int); values=test[score].fillna(0)
        precision,recall,_=precision_recall_curve(y,values)
        fig,ax=plt.subplots(figsize=(6,4)); ax.plot(recall,precision); ax.set(xlabel="Recall",ylabel="Precision",title=f"{title} PR — real held-out (n={len(test)})")
        fig.tight_layout(); fig.savefig(plot_dir/f"{score}_precision_recall.png",dpi=140); plt.close(fig)
    available=pd.Series({c.removesuffix("_available"):test[c].fillna(False).mean() for c in test if c.endswith("_available")}).sort_values()
    fig,ax=plt.subplots(figsize=(8,5)); available.plot.barh(ax=ax); ax.set(xlim=(0,1),xlabel="Available fraction",title=f"Diagnostic availability — real held-out (n={len(test)})")
    fig.tight_layout(); fig.savefig(plot_dir/"diagnostic_availability.png",dpi=140); plt.close(fig)

def promotion_gates(result: dict, phase1_status: str) -> tuple[str,list[str]]:
    failures=[]; eb=result["eb"]; blend=result["blend"]
    if phase1_status!="PASS": failures.append(f"Phase 1 status is {phase1_status}")
    if eb["recall"]<0.90: failures.append(f"EB recall {eb['tp']}/{eb['positive_support']} < 0.90")
    if blend["recall"]<0.80: failures.append(f"blend recall {blend['tp']}/{blend['positive_support']} < 0.80")
    if blend["precision"]<0.75: failures.append(f"blend precision {blend['tp']}/{blend['tp']+blend['fp']} < 0.75")
    if result["clean_planet_false_blend_rate"] is None or result["clean_planet_false_blend_rate"]>0.10:
        failures.append("clean-planet false blend-flag rate exceeds 0.10 or is unavailable")
    minimum={"exoplanet_transit":200,"eclipsing_binary":200,"blend_contamination":100}
    supports={"exoplanet_transit":result["clean_planet_support"],"eclipsing_binary":eb["positive_support"],"blend_contamination":blend["positive_support"]}
    for name,needed in minimum.items():
        if supports[name]<needed: failures.append(f"blind support for {name} is {supports[name]} < {needed}")
    return ("PASS" if not failures else "PARTIAL"),failures
