import os
import glob
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
INPUT_FOLDER = r"D:\TEST OUPUTS\results"
OUTPUT_FILE = r"D:\TEST OUPUTS\results\tournament_master_log.csv"
# ---------------------

def consolidate():
    all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    # Exclude the master log itself
    all_files = [f for f in all_files if "tournament_master_log" not in f]
    
    print(f"Found {len(all_files)} CSV files to merge.")

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # --- NEW: Add the filename as a column so we can identify the video later ---
            # We take just the stem (e.g., "match_0000" from "match_0000.csv")
            df['source_video'] = Path(filename).stem
            # -----------------------------------------------------------------------
            df_list.append(df)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if df_list:
        master_df = pd.concat(df_list, axis=0, ignore_index=True)
        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✔ Success! Merged with filenames into: {OUTPUT_FILE}")
    else:
        print("Could not merge files.")

if __name__ == "__main__":
    consolidate()