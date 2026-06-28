import os
import sys
import pandas as pd
from pathlib import Path

# Add pipeline to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phase1.config import Config
from scratch.generate_mock_dataset import compute_sha256

def main():
    config = Config()
    manifests_dir = config.manifests_dir
    download_manifest_path = manifests_dir / "download_manifest.parquet"
    discovery_manifest_path = manifests_dir / "discovery_manifest.parquet"
    
    try:
        df_dl = pd.read_parquet(download_manifest_path)
        print("Successfully read existing download manifest.")
    except Exception as e:
        print(f"Download manifest corrupted or missing ({e}). Rebuilding from discovery manifest...")
        if not discovery_manifest_path.exists():
            print("Error: discovery_manifest.parquet not found either.")
            sys.exit(1)
        df_dl = pd.read_parquet(discovery_manifest_path)
        df_dl["local_path"] = ""
        df_dl["actual_size"] = 0
        df_dl["sha256"] = ""
        df_dl["attempt_count"] = 0
        df_dl["final_status"] = "pending"
        df_dl["failure_message"] = ""
        
    print("Initial status counts:")
    print(df_dl["final_status"].value_counts())
    
    # Reset final_status for all rows where FITS file exists and is readable
    count = 0
    for idx, row in df_dl.iterrows():
        sec = int(row["sector"])
        dest_file = config.raw_dir / f"sector_{sec:04d}" / "lightcurves" / row["product_filename"]
        
        if dest_file.exists():
            # If not already verified, verify it
            if df_dl.at[idx, "final_status"] != "verified" and df_dl.at[idx, "final_status"] != "processed":
                try:
                    sha = compute_sha256(dest_file)
                    size = dest_file.stat().st_size
                    
                    df_dl.at[idx, "final_status"] = "verified"
                    df_dl.at[idx, "local_path"] = str(dest_file)
                    df_dl.at[idx, "sha256"] = sha
                    df_dl.at[idx, "actual_size"] = size
                    df_dl.at[idx, "attempt_count"] = 1
                    df_dl.at[idx, "failure_message"] = ""
                    count += 1
                except Exception as e:
                    print(f"Error checking file for TIC {row['tic_id']}: {e}")
                    
    df_dl.to_parquet(download_manifest_path, index=False)
    print(f"Updated {count} rows to verified status based on existing raw files.")
    print("New status counts:")
    print(df_dl["final_status"].value_counts())

if __name__ == "__main__":
    main()
