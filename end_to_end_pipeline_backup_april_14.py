import luigi
import os
import subprocess
import glob
import pandas as pd
import sys
import time
import re
import json
from datetime import datetime, timedelta
from pathlib import Path

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
BASE_DIR = r"D:\Judo_Pipeline"
RAW_DIR = os.path.join(BASE_DIR, "01_Raw_Input")
CONVERTED_DIR = os.path.join(BASE_DIR, "02_Converted")
SEGMENTED_DIR = os.path.join(BASE_DIR, "03_Segmented")
FRAMES_DIR = os.path.join(BASE_DIR, "04_Frames")
RESULTS_DIR = os.path.join(BASE_DIR, "05_Results")
FINAL_CLIPS_DIR = os.path.join(BASE_DIR, "06_Final_Clips")

PROJECT_JSON = os.path.join(RESULTS_DIR, "project_manifest.json")
MASTER_CSV = os.path.join(RESULTS_DIR, "tournament_master_log.csv")

# Ensure base directories exist
for d in [RAW_DIR, CONVERTED_DIR, SEGMENTED_DIR, FRAMES_DIR, RESULTS_DIR, FINAL_CLIPS_DIR]:
    os.makedirs(d, exist_ok=True)

def get_raw_videos():
    """Helper function to count inputs and trigger dynamic updates."""
    return list(Path(RAW_DIR).glob("*.*"))

# --- NEW: Global dictionary to store our task times ---
TASK_TIMINGS = {}

# ==========================================
# 1.5 GLOBAL TASK TIMERS (Luigi Event Handlers)
# ==========================================

@luigi.Task.event_handler(luigi.Event.START)
def start_timing(task):
    """Starts the stopwatch when a task begins."""
    task._start_time = time.time()

@luigi.Task.event_handler(luigi.Event.SUCCESS)
def success_timing(task):
    """Stops the stopwatch, saves, and prints the duration when a task succeeds."""
    if hasattr(task, '_start_time'):
        elapsed_seconds = time.time() - task._start_time
        mins, secs = divmod(elapsed_seconds, 60)
        time_str = f"{int(mins)}m {secs:.1f}s"
        
        # NEW LOGIC: Store the time in our global ledger
        TASK_TIMINGS[task.__class__.__name__] = time_str
        
        print(f"\n[⏱️ TIMER] {task.__class__.__name__} completed in {time_str}")

@luigi.Task.event_handler(luigi.Event.FAILURE)
def failure_timing(task, exception):
    """Stops the stopwatch and prints the duration if a task crashes."""
    if hasattr(task, '_start_time'):
        elapsed_seconds = time.time() - task._start_time
        mins, secs = divmod(elapsed_seconds, 60)
        print(f"\n[⚠️ TIMER] {task.__class__.__name__} FAILED after {int(mins)}m {secs:.1f}s")

# ==========================================
# 2. LUIGI PIPELINE TASKS
# ==========================================

class Task1_FormatVideo(luigi.Task):
    """Formats a SINGLE video. Skips if this specific MP4 already exists."""
    video_path = luigi.Parameter()

    def output(self):
        video_stem = Path(str(self.video_path)).stem
        out_name = os.path.join(CONVERTED_DIR, f"{video_stem}_std.mp4")
        return luigi.LocalTarget(out_name)

    def run(self):
        print(f"\n>>> TASK 1: Formatting {self.video_path}...")
        cmd = [
            "ffmpeg", "-y", "-i", str(self.video_path), 
            "-vsync", "1",               # FORCE Constant Frame Rate
            "-r", "30",                  # Set framerate to exactly 30
            "-c:v", "libx264", 
            "-preset", "fast", "-crf", "22", 
            "-af", "aresample=async=1",  # Keep audio perfectly synced with new video timing
            self.output().path
        ]
        subprocess.run(cmd, check=True)


class Task2_SegmentVideos(luigi.Task):
    """Batch processes the folder, but re-runs if the raw video count changes."""
    def requires(self):
        # Demand Task 1 completes for EVERY file in the raw folder
        raw_files = get_raw_videos()
        if not raw_files:
            raise FileNotFoundError(f"Drop raw videos into {RAW_DIR} first!")
        return [Task1_FormatVideo(video_path=str(v)) for v in raw_files]

    def output(self):
        num_vids = len(get_raw_videos())
        return luigi.LocalTarget(os.path.join(SEGMENTED_DIR, f"_SEGMENTED_{num_vids}_FILES"))

    def run(self):
        print("\n>>> TASK 2: Segmenting Videos...")
        subprocess.run([
            sys.executable, "-m", "judo_footage_analysis.workflow.truncate_videos",
            "--input-root-path", CONVERTED_DIR,
            "--output-root-path", SEGMENTED_DIR
        ], check=True)
        with self.output().open('w') as f: f.write("Done")


class Task3_ExtractFrames(luigi.Task):
    """Bypasses module errors and forces FFmpeg to extract frames from segments."""
    def requires(self): return Task2_SegmentVideos()
    
    def output(self): 
        num_vids = len(get_raw_videos())
        return luigi.LocalTarget(os.path.join(FRAMES_DIR, f"_FRAMES_{num_vids}_FILES"))

    def run(self):
        print("\n>>> TASK 3: Extracting Frames...")
        segments = list(Path(SEGMENTED_DIR).rglob("*.mp4"))
        for video in segments:
            target_folder = os.path.join(FRAMES_DIR, f"{video.parent.name}_{video.stem}")
            os.makedirs(target_folder, exist_ok=True)
            if len(os.listdir(target_folder)) < 100: # Skip if already processed
                cmd = ["ffmpeg", "-y", "-i", str(video), "-vf", "fps=30", "-q:v", "5", os.path.join(target_folder, "%06d.jpg")]
                subprocess.run(cmd, check=True)
        with self.output().open('w') as f: f.write("Done")


class Task4_GenerateManifest(luigi.Task):
    """Creates the JSON map required by the YOLO AI."""
    def requires(self): return Task3_ExtractFrames()
    
    def output(self): 
        num_vids = len(get_raw_videos())
        return luigi.LocalTarget(os.path.join(RESULTS_DIR, f"_MANIFEST_{num_vids}_FILES"))

    def run(self):
        print("\n>>> TASK 4: Generating AI Manifest...")
        subprocess.run([
            sys.executable, "scripts/generate_combat_json.py",
            "--input_folder", SEGMENTED_DIR,
            "--output_path", PROJECT_JSON
        ], check=True)
        with self.output().open('w') as f: f.write("Done")


class Task5_RunAIAnalysis(luigi.Task):
    """Executes the YOLOv8 model to classify combat phases."""
    def requires(self): return Task4_GenerateManifest()
    
    def output(self): 
        num_vids = len(get_raw_videos())
        return luigi.LocalTarget(os.path.join(RESULTS_DIR, f"_AI_{num_vids}_FILES"))

    def run(self):
        print("\n>>> TASK 5: Running AI Analysis...")
        subprocess.run([
            sys.executable, "-m", "judo_footage_analysis.workflow.extract_combat_phases", "ExtractCombatPhases",
            "--project-json", PROJECT_JSON,
            "--output-dir", RESULTS_DIR,
            "--local-scheduler"
        ], check=True)
        with self.output().open('w') as f: f.write("Done")


class Task6_ConsolidateAndClip(luigi.Task):
    """Merges the AI data and uses FFmpeg to trim the final action clips."""
    def requires(self): return Task5_RunAIAnalysis()
    
    def output(self): 
        num_vids = len(get_raw_videos())
        return luigi.LocalTarget(os.path.join(FINAL_CLIPS_DIR, f"_PIPELINE_COMPLETE_{num_vids}_FILES"))

    def run(self):
        print("\n>>> TASK 6: Consolidating Data & Clipping Matches...")
        
        all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
        all_files = [f for f in all_files if "tournament_master_log" not in f]
        
        df_list = []
        for file in all_files:
            df = pd.read_csv(file)
            clean_name = Path(file).stem.replace(".mp4_phases", "")
            df['source_video'] = clean_name 
            df_list.append(df)
            
        if not df_list:
            raise ValueError("No CSV data found to process!")
            
        master_df = pd.concat(df_list, axis=0, ignore_index=True)
        master_df.to_csv(MASTER_CSV, index=False)

# 2. Clip the Matches in Contiguous Blocks
        grouped = master_df.groupby('source_video')
        for video_name, data in grouped:
            data = data.sort_values('timestamp')
            
            # Clean the text
            clean_phases = data['phase'].astype(str).str.strip().str.lower()
            inactive_phases = ['mate', 'no-match/intermission', 'none', 'nan']
            
            # NEW LOGIC: Must be an active phase AND have at least 2 bounding box detections
            is_active_phase = ~clean_phases.isin(inactive_phases)
            has_fighters = data['detections'] >= 2
            
           # Combine the two rules
            is_action = is_active_phase & has_fighters
            
            # THE PATIENCE BUFFER: Bridge gaps of up to 90 frames (3 seconds at 30fps)
            is_action = is_action.replace(False, pd.NA).ffill(limit=90).fillna(False).astype(bool)
            
          # NEW LOGIC: Reverse-Cooldown Debouncer + Crowd Density Filter
            raw_bows = data.get('bow_detected', pd.Series(False, index=data.index)).fillna(False).astype(bool)
            
            # CROWD FILTER: Ignore any bow hallucinated while the mat is flooded with people (e.g., > 8 detections)
            is_not_crowded = data['detections'] <= 8
            bows = raw_bows & is_not_crowded
            
            # Isolate the exact timestamps where VALID bows occurred
            bow_timestamps = data.loc[bows, 'timestamp']
            # Calculate the time difference to the NEXT bow. 
            # We only keep a bow if the next bow is MORE than 15 seconds away (or if it's the final bow of the day).
            # This mathematically deletes the edge-mat bow and keeps the center-mat bow.
            time_to_next_bow = bow_timestamps.diff(-1).abs()
            valid_bows_mask = (time_to_next_bow > 15) | (time_to_next_bow.isna())
            
            # Apply the filtered bows back to the main dataset
            data['valid_bow'] = False
            data.loc[bow_timestamps[valid_bows_mask].index, 'valid_bow'] = True
            
            # Increment match ID only on the validated bows
            data['match_id'] = data['valid_bow'].cumsum()
            
            # Define block IDs by action changes OR match_id changes (forces a cut on a bow)
            block_changes = (is_action != is_action.shift()) | (data['match_id'] != data['match_id'].shift())
            block_ids = block_changes.cumsum()
            
            active_blocks = data[is_action].groupby(block_ids)
            
            source_vids = list(Path(SEGMENTED_DIR).rglob(f"*{video_name}*.mp4"))
            if not source_vids:
                continue
                
            clip_count = 1
            for block_id, block_data in active_blocks:
                # INCREASED blip filter from 3 seconds to 5 seconds to kill random noise
                if len(block_data) < 5: 
                    continue
                    
                start_time = max(0, block_data['timestamp'].min() - 2)
                end_time = block_data['timestamp'].max() + 2
                
                # --- NEW LOGIC: HIERARCHICAL TIME EXTRACTION ---
                base_dt = None
                
                # Grab the full hard drive path to the video
                full_source_path = str(source_vids[0])
                
                # Create a clean parent name so final clips from different matches don't overwrite each other
                parent_folder = source_vids[0].parent.name.replace("match_", "").replace("_std", "")
                clip_base_name = f"{parent_folder}_part_{video_name}"
                
                # Priority 1: Check the full path string (not just the chunk name)
                match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})", full_source_path)
                if match:
                    base_dt = datetime.strptime(match.group(1), "%Y-%m-%d_%H_%M_%S")
                else:
                    # Priority 2: Extract internal metadata from the RAW file
                    try:
                        raw_files = list(Path(RAW_DIR).glob(f"*{parent_folder}*.*"))
                        if raw_files:
                            cmd = [
                                "ffprobe", "-v", "quiet", 
                                "-show_entries", "format_tags=creation_time", 
                                "-of", "default=noprint_wrappers=1:nokey=1", 
                                str(raw_files[0])
                            ]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            creation_str = result.stdout.strip()
                            
                            if creation_str:
                                clean_time = creation_str.split('.')[0].replace('T', ' ').replace('Z', '')
                                base_dt = datetime.strptime(clean_time, "%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        pass
                
                # Apply the extracted base time or fallback
                if base_dt:
                    # --- NEW LOGIC: Calculate the missing hours ---
                    # The segmenter cuts the video into 10-minute (600 second) chunks. Change value below if clips are different length.
                    # We multiply the chunk name (e.g., '0053') by 600 to find the true offset.
                    try:
                        chunk_index = int(video_name)
                        chunk_offset_seconds = chunk_index * 600
                    except ValueError:
                        chunk_offset_seconds = 0
                    
                    # Add the previous chunks AND the current clip's start time to the morning base time
                    total_elapsed_seconds = chunk_offset_seconds + start_time
                    clip_dt = base_dt + timedelta(seconds=total_elapsed_seconds)
                    time_str = clip_dt.strftime("%Hh%Mm%Ss")
                else:
                    # Priority 3: Final Fallback (elapsed seconds)
                    hrs, rem = divmod(int(start_time), 3600)
                    mins, secs = divmod(rem, 60)
                    time_str = f"elapsed_{hrs:02d}h{mins:02d}m{secs:02d}s"
                
                out_path = os.path.join(FINAL_CLIPS_DIR, f"{clip_base_name}_AT_{time_str}.mp4")
                print(f"Clipping {clip_base_name} -> {start_time:.1f}s to {end_time:.1f}s (Saved as {clip_base_name}_AT_{time_str}.mp4)")
                
                cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time), "-i", str(source_vids[0]), out_path]
                subprocess.run(cmd, check=True)
                
                clip_count += 1
                
        with self.output().open('w') as f: f.write("Done")

# ==========================================
# 3. TRIGGER
# ==========================================
if __name__ == "__main__":
    print("\n>>> STARTING JUDO PIPELINE <<<")
    start_time = time.time()
    
    # Pointing Luigi to the final task causes it to chain everything else automatically
    luigi.build([Task6_ConsolidateAndClip()], workers=1, local_scheduler=False)
    
    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    
    # --- NEW LOGIC: Print the Final Summary Report ---
    print("\n" + "="*50)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("="*50)
    
    if not TASK_TIMINGS:
        print(" No new tasks were executed (all outputs already exist).")
    else:
        for task_name, duration in TASK_TIMINGS.items():
            # This formatting pads the task name with spaces so all the times align perfectly
            print(f" {task_name.ljust(25)} : {duration}")
            
    print("-" * 50)
    print(f" TOTAL EXECUTION TIME      : {total_minutes:.2f} minutes")
    print("="*50 + "\n")