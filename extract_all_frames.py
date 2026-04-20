import os
from pathlib import Path
from judo_footage_analysis.frame_extraction import extract_frames

# --- CONFIGURATION ---
# Path where your segmented clips are located (from the previous step)
INPUT_DIR = r"D:\TEST OUPUTS\Truncated\match__test_output_1"

# Path where you want the images to be saved
OUTPUT_DIR = r"D:\TEST OUPUTS\IMAGES"
# ---------------------

def run_extraction():
    # Convert strings to Path objects
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .mp4 files in the segmented matches folder
    # Note: The segmenter likely created subfolders (e.g. match_video1/0000.mp4)
    # rglob('*') finds all files recursively
    video_files = list(input_path.rglob("*.mp4"))
    
    print(f"Found {len(video_files)} video segments to process.")

    for video in video_files:
        # Create a specific subfolder for this video's frames to keep things organized
        # e.g. .../extracted_frames/match_video1_0000/
        relative_name = f"{video.parent.name}_{video.stem}"
        video_output_folder = output_path / relative_name
        
        print(f"Extracting frames for: {video.name}")
        
        try:
            # Call the function specified in Figure 5 of the documentation
            extract_frames(str(video), str(video_output_folder))
            print(f"Done. Saved to: {video_output_folder}")
        except Exception as e:
            print(f"Failed to extract {video.name}: {e}")

if __name__ == "__main__":
    run_extraction()