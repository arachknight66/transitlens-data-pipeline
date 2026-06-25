import argparse
import pandas as pd
import sys
import os

# Add the local directory to sys.path so we can import from the pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interface import load_light_curve

def main():
    parser = argparse.ArgumentParser(description="Download a real TESS light curve and save as CSV for TransitLens platform.")
    parser.add_argument("tic_id", type=str, help="The TIC ID to download (e.g. 261136679)")
    parser.add_argument("--output", "-o", type=str, help="Output CSV file path. Defaults to <tic_id>.csv")
    
    args = parser.parse_args()
    tic_id = args.tic_id
    output_file = args.output if args.output else f"{tic_id.replace('TIC', '').replace('-', '').strip()}.csv"

    print(f"Fetching TESS data for TIC ID: {tic_id}...")
    try:
        # This uses the Phase 5 stretch-goal path which caches to real_tess/cache
        result = load_light_curve("tess", tic_id)
    except ImportError as e:
        print(f"ERROR: {e}")
        print("\nPlease install the required dependencies: pip install lightkurve astroquery")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR fetching data: {e}")
        sys.exit(1)

    time_data = result["time"]
    flux_data = result["flux"]

    print(f"Successfully retrieved {len(time_data)} data points.")
    
    df = pd.DataFrame({
        "time": time_data,
        "flux": flux_data
    })
    
    # Drop rows with NaN in either time or flux
    df = df.dropna(subset=["time", "flux"])
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data to {len(df)} non-NaN data points.")
    print(f"Saved light curve to {output_file}")
    print("You can now upload this file to the TransitLens platform.")

if __name__ == "__main__":
    main()
