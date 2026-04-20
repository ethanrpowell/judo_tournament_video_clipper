#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

def generate_json(video_folder, output_json_path):
    if not os.path.exists(video_folder):
        raise FileNotFoundError(f"Video folder not found: {video_folder}")

    # Recursively find all mp4s in case they are nested in subfolders
    videos = list(Path(video_folder).rglob("*.mp4"))

    if not videos:
        print(f"No MP4 files found in {video_folder}")
        return

    data = [{"video": str(v), "annotations": []} for v in videos]

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"JSON file created with {len(videos)} videos at: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    generate_json(args.input_folder, args.output_path)