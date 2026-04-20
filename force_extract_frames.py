import os
import subprocess
from pathlib import Path
import imageio_ffmpeg

# --- CONFIGURATION ---
INPUT_DIR = r"D:\TEST OUPUTS\Truncated"
OUTPUT_DIR = r"D:\TEST OUPUTS\frames"
# ---------------------

def run_extraction():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    input_path = Path(INPUT_DIR)
    
    video_files = list(input_path.rglob("*.mp4"))
    print(f"Found {len(video_files)} video segments.")

    for video in video_files:
        # Create output folder
        subfolder_name = f"{video.parent.name}_{video.stem}"
        target_folder = os.path.join(OUTPUT_DIR, subfolder_name)
        os.makedirs(target_folder, exist_ok=True)
        
        # SKIP CHECK: If folder already has >100 images, assume it's done.
        # This saves you from re-doing the first few matches.
        existing_files = len(os.listdir(target_folder))
        if existing_files > 100:
            print(f"Skipping {video.name} (Found {existing_files} frames already).")
            continue

        print(f"Extracting: {video.name}...")

        cmd = [
            ffmpeg_exe,
            "-y",               # <--- FIX: Forces overwrite (No more hanging)
            "-i", str(video),
            "-vf", "fps=30", 
            "-q:v", "2", 
            os.path.join(target_folder, "%06d.jpg")
        ]

        try:
            # FIX: Removed 'capture_output=True' so you can see the progress scrolling
            subprocess.run(cmd, check=True) 
            print(f"✔ Success: {video.name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {video.name} - {e}")

if __name__ == "__main__":
    run_extraction()