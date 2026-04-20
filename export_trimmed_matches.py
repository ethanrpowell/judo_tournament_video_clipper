import pandas as pd
import subprocess
import os
from pathlib import Path

# --- CONFIGURATION ---
# The master log we just created
CSV_PATH = r"D:\TEST OUPUTS\results\tournament_master_log.csv"

# Where the original videos are (MP4s)
VIDEO_SOURCE_DIR = r"D:\TEST OUPUTS\Truncated"

# Where to save the new trimmed clips
OUTPUT_CLIPS_DIR = r"D:\TEST OUPUTS\Final_Clips"
# ---------------------

def extract_clips():
    # 1. Load the data
    if not os.path.exists(CSV_PATH):
        print("CSV not found. Please run consolidate_results.py first.")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)

    # 2. Group data by video file
    # We look at each video individually to find its specific start/end
    grouped = df.groupby('source_video')

    print(f"Found {len(grouped)} matches to process...")

    for video_name, data in grouped:
        # 3. Find Start and End Times
        # We assume 'Action' is anything that is NOT 'Mate' (Paused)
        # If your phase labels are "Tachi-waza", "Ne-waza", "Mate", this works.
        
        # Filter for rows where the match is active
        active_data = data[data['phase'] != 'Mate']
        
        if active_data.empty:
            print(f"Skipping {video_name}: No action detected.")
            continue

        # Get the first and last frame of action
        # Assuming the CSV has a 'frame' or 'timestamp' column. 
        # If it's index-based, we use the index.
        start_frame = active_data.index.min() # Relative to the original DF, might be risky.
        # Better: Use the 'frame' column if it exists, otherwise assume row count
        
        if 'frame' in active_data.columns:
            start_frame = active_data['frame'].min()
            end_frame = active_data['frame'].max()
        else:
            # Fallback: assuming rows are seconds or sequential frames
            # This logic depends on your exact CSV structure. 
            # Let's assume 30fps for calculation if 'timestamp' is missing.
            start_frame = active_data.index.min() - data.index.min() 
            end_frame = active_data.index.max() - data.index.min()

        # Convert frames to Seconds (Assuming 30 FPS - ADJUST IF NEEDED)
        fps = 30.0
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # Add a small buffer (e.g., 2 seconds) to not cut abruptly
        start_time = max(0, start_time - 2)
        end_time = end_time + 2

        # 4. Construct Paths
        # We need to find the original MP4. It might be in a subfolder.
        # We search recursively for the filename.
        found_videos = list(Path(VIDEO_SOURCE_DIR).rglob(f"{video_name}.mp4"))
        
        if not found_videos:
            # Try matching partial names if the CSV name is slightly different
            # (e.g. CSV has 'match_0000', video is '0000.mp4')
            # This is a common issue, so we try a fallback search:
            found_videos = list(Path(VIDEO_SOURCE_DIR).rglob(f"*{video_name}*.mp4"))
            
        if not found_videos:
            print(f"❌ Could not find video source for: {video_name}")
            continue
            
        input_video_path = str(found_videos[0])
        output_clip_path = os.path.join(OUTPUT_CLIPS_DIR, f"{video_name}_trimmed.mp4")

        print(f"✂ Clipping {video_name}: {start_time:.1f}s to {end_time:.1f}s")

        # 5. Run FFmpeg to cut without re-encoding (Fast!)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),       # Start timestamp
            "-to", str(end_time),         # End timestamp
            "-i", input_video_path,       # Input
            "-c", "copy",                 # Copy stream (no quality loss, very fast)
            output_clip_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"   ✔ Saved to: {output_clip_path}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ FFmpeg Error: {e}")

if __name__ == "__main__":
    extract_clips()