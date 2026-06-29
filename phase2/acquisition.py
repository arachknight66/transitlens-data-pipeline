"""Budgeted, resumable TPF acquisition for explicitly scoped Phase 2 targets."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, shutil, threading, time, urllib.request
import pandas as pd

from phase2.tpf_downloader import TpfDownloader

def build_acquisition_manifest(config, splits=("val","test")) -> pd.DataFrame:
    m=config.manifests_dir
    names={"val":"phase2_features_validation.parquet","test":"phase2_features_test.parquet","train":"phase2_features_train.parquet"}
    features=pd.concat([pd.read_parquet(m/names[s])[["tic_id","observation_id","sector","split"]] for s in splits],ignore_index=True)
    benchmark=pd.read_parquet(m/"phase2_benchmark_manifest.parquet")
    products=benchmark[["observation_id","product_uri"]].drop_duplicates("observation_id")
    result=features.merge(products,on="observation_id",how="left",validate="one_to_one")
    result["tpf_uri"]=result.product_uri.str.replace("_lc.fits","_tp.fits",regex=False)
    result["tpf_filename"]=result.tpf_uri.str.rsplit("/",n=1).str[-1]
    result["expected_bytes"]=pd.Series([pd.NA]*len(result),dtype="Int64")
    result["local_path"]=[str(config.tpf_dir/name) for name in result.tpf_filename]
    result["status"]="pending"; result["error"]=""
    return result

def spatial_critical_subset(config, acquisition: pd.DataFrame) -> pd.DataFrame:
    """Detected planet/blend targets that determine blend gates and false flags."""
    m=config.manifests_dir
    metadata=pd.read_parquet(m/"phase2_feature_metadata.parquet")[["observation_id","canonical_label"]]
    features=pd.concat([pd.read_parquet(m/"phase2_features_validation.parquet"),
                        pd.read_parquet(m/"phase2_features_test.parquet")],ignore_index=True)
    useful=features.loc[features.candidate_detected.fillna(False),["observation_id"]].merge(metadata,on="observation_id")
    useful=useful[useful.canonical_label.isin(["blend_contamination","exoplanet_transit"])]
    return acquisition[acquisition.observation_id.isin(set(useful.observation_id))].copy()

def _remote_size(uri: str) -> int:
    url="https://mast.stsci.edu/api/v0.1/Download/file?uri="+uri
    request=urllib.request.Request(url,headers={"Range":"bytes=0-0","User-Agent":"TransitLens/2.2"})
    with urllib.request.urlopen(request,timeout=60) as response:
        content_range=response.headers.get("Content-Range",""); response.read(1)
        return int(content_range.split("/")[-1]) if "/" in content_range else int(response.headers.get("Content-Length") or 0)

def estimate_manifest_sizes(frame: pd.DataFrame, workers: int=12) -> pd.DataFrame:
    output=frame.copy()
    def measure(item):
        index,uri=item
        try:return index,_remote_size(uri),""
        except Exception as exc:return index,0,f"{type(exc).__name__}: {exc}"
    with ThreadPoolExecutor(max_workers=min(workers,16)) as pool:
        for index,size,error in pool.map(measure,output.tpf_uri.items()):
            output.at[index,"expected_bytes"]=size if size else pd.NA
            if error: output.at[index,"error"]=error; output.at[index,"status"]="size_query_failed"
    return output

def download_manifest(config, frame: pd.DataFrame, *, workers: int=6, reserve_bytes: int=20_000_000_000,
                      progress_path: Path|None=None) -> dict:
    expected=int(frame.expected_bytes.fillna(0).sum())
    free=shutil.disk_usage(config.tpf_dir).free
    if expected and free-expected<reserve_bytes:
        raise RuntimeError(f"TPFs require {expected/1e9:.1f} GB but safety reserve would be violated (free {free/1e9:.1f} GB)")
    downloader=TpfDownloader(config.tpf_dir); lock=threading.Lock(); records=[]
    def one(row):
        result=downloader.download_tpf(int(row.tic_id),int(row.sector),str(row.tpf_uri))
        result["observation_id"]=row.observation_id; result["expected_bytes"]=None if pd.isna(row.expected_bytes) else int(row.expected_bytes)
        return result
    with ThreadPoolExecutor(max_workers=min(workers,8)) as pool:
        futures=[pool.submit(one,row) for row in frame.itertuples(index=False)]
        for future in as_completed(futures):
            result=future.result()
            with lock:
                records.append(result)
                if progress_path and len(records)%10==0:
                    progress_path.write_text(json.dumps({"completed":len(records),"total":len(frame),"results":records},indent=2),encoding="utf-8")
    result_frame=pd.DataFrame(records)
    result_frame.to_parquet(config.manifests_dir/"tpf_acquisition_manifest.parquet",index=False)
    verified=int(result_frame.status.eq("verified").sum())
    summary={"targets":len(frame),"verified":verified,
             "failed":int(len(result_frame)-verified),"expected_gb":expected/1e9,
             "downloaded_gb":sum(Path(p).stat().st_size for p in result_frame.local_path if Path(p).exists())/1e9}
    if progress_path: progress_path.write_text(json.dumps({**summary,"results":records},indent=2),encoding="utf-8")
    return summary
