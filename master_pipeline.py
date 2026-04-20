import luigi
import subprocess
import os
from pathlib import Path

# --- CONFIGURATION ---
RAW_VIDEO_FOLDER = r"C:\Users\e3powell\Desktop\raw_videos"        # Put your 4hr raw files here
SEGMENTED_FOLDER = r"C:\Users\e3powell\Desktop\segmented_matches" # Where clips go
PROJECT_JSON = r"data\combat_phase\project.json"
RESULTS_DIR = r"data\combat_phase\results"
FINAL_REPORT = r"data\combat_phase\final_report.txt"
# ---------------------

class Step1_SegmentVideos(luigi.Task):
    """
    Runs the segmentation script to chop raw videos into matches.
    """
    def output(self):
        # We assume if the folder exists and has content, we are done.
        # Ideally, we'd check for a specific 'done' flag file.
        return luigi.LocalTarget(os.path.join(SEGMENTED_FOLDER, "_SEGMENTATION_COMPLETE"))

    def run(self):
        print(">>> STEP 1: SEGMENTING VIDEOS...")
        
        # Ensure raw folder exists
        if not os.path.exists(RAW_VIDEO_FOLDER):
            raise FileNotFoundError(f"Please create {RAW_VIDEO_FOLDER} and put your raw videos there!")

        # Call your existing segmentation logic (using the command line tool)
        # Note: We use 'python -m' to run the module we set up earlier
        subprocess.run([
            "python", "-m", "judo_footage_analysis.workflow.truncate_videos",
            "--input_path", RAW_VIDEO_FOLDER,
            "--output_path", SEGMENTED_FOLDER,
            "--local_scheduler" # Run this part locally
        ], check=True)

        # Mark as complete
        with self.output().open('w') as f:
            f.write("Segmentation Done")

class Step2_GenerateManifest(luigi.Task):
    """
    Scans the new segments and builds the project.json map.
    """
    def requires(self):
        return Step1_SegmentVideos()

    def output(self):
        return luigi.LocalTarget(PROJECT_JSON)

    def run(self):
        print(">>> STEP 2: GENERATING JSON MAP...")
        
        os.makedirs(os.path.dirname(PROJECT_JSON), exist_ok=True)
        
        subprocess.run([
            "python", "scripts/generate_combat_json.py",
            "--input_folder", SEGMENTED_FOLDER,
            "--output_path", PROJECT_JSON
        ], check=True)

class Step3_RunAIAnalysis(luigi.Task):
    """
    Runs the YOLO AI analysis on the matches found in the JSON.
    """
    def requires(self):
        return Step2_GenerateManifest()

    def output(self):
        # Checks if the final consolidated CSV exists
        return luigi.LocalTarget(os.path.join(RESULTS_DIR, "tournament_master_log.csv"))

    def run(self):
        print(">>> STEP 3: RUNNING AI ANALYSIS (This may take a while)...")
        
        # 1. Run the extraction workflow
        subprocess.run([
            "python", "-m", "judo_footage_analysis.workflow.extract_combat_phases",
            "ExtractCombatPhases",
            "--project-json", PROJECT_JSON,
            "--output-dir", RESULTS_DIR,
            "--local-scheduler"
        ], check=True)

        # 2. Consolidate the results (Merging the CSVs)
        # We run a small inline script to merge them right here
        print(">>> Consolidating CSV files...")
        import pandas as pd
        import glob
        
        all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
        df_list = [pd.read_csv(f) for f in all_files if "tournament_master_log" not in f]
        
        if df_list:
            master_df = pd.concat(df_list, ignore_index=True)
            master_df.to_csv(self.output().path, index=False)
        else:
            # Create empty file if no results, just to satisfy Luigi
            open(self.output().path, 'a').close()

class Step4_FinalReport(luigi.Task):
    """
    Generates the final text report for the coach.
    """
    def requires(self):
        return Step3_RunAIAnalysis()

    def output(self):
        return luigi.LocalTarget(FINAL_REPORT)

    def run(self):
        print(">>> STEP 4: GENERATING COACH REPORT...")
        
        # Call the report script we wrote earlier
        # (Assuming you saved it as generate_match_report.py)
        # We pass the input/output paths as arguments or set env vars, 
        # but for simplicity, we'll just run the script and move the file.
        
        cmd = ["python", "generate_match_report.py"]
        
        # Capture the output of the script and write it to the final file
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        with self.output().open('w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n--- ERRORS ---\n")
                f.write(result.stderr)

# --- THE TRIGGER ---
if __name__ == "__main__":
    # This builds the final step, which triggers all previous steps automatically
    luigi.build([Step4_FinalReport()], local_scheduler=True)
