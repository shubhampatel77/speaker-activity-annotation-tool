import cv2
import json
import os
import tqdm
import time
import sys
sys.path.append("../goturn-pytorch/src")
from pathlib import Path
from check_videos import check_videos_in_directory
from track import process_video
from utils import load_goturn_model, add_audio_to_video, check_audio_stream



def main():
    original_videos_path = '../data/original_videos'
    tracked_videos_path = '../data/annotations/tracked_videos'
    tracks_path = '../data/annotations/tracks'
    
    os.makedirs(tracks_path, exist_ok=True)
    os.makedirs(tracked_videos_path, exist_ok=True)
    
    # iou_threshold, fallback_threshold, min_track_len
    thresholds = 0.2, 0.2, 5
    
    ckpt_path = "../goturn-pytorch/src/goturn/models/pretrained/_ckpt_epoch_3.ckpt"
    goturn_model = load_goturn_model(ckpt_path)
    
    # Step 1: Check if the videos in the original_videos_path folder are valid
    print("\nChecking original videos...")
    successful, failed = check_videos_in_directory(original_videos_path)
    
    # print("\nSuccessfully loaded video paths:")
    # for path in successful:
    #     print(path)

    print("\nFailed to load video paths:")
    for path in failed:
        print(path)
        
    # Step 2: Process each successful video and annotate them
    print("\nProcessing each video for annotation and generating JSON files...")
    for filename in successful:
        filename = Path(filename).stem
        if filename == "4L5Yv6gADuM":
            input_video_path = f'../data/original_videos/{filename}.mp4'
            output_json_path = f'../data/annotations/tracks/{filename}.json'

            # Check if output files already exist
            if os.path.exists(output_json_path):
                print(f"Skipping {filename} as output files already exist.")
                continue

            process_video(input_video_path, output_json_path, goturn_model, thresholds)
            assert 1==2

    print("\nAll videos processed successfully for annotation!")
    
    filename = "4L5Yv6gADuM"
    original_video_path = f'../data/original_videos/{filename}.mp4'
    json_path = f'../data/annotations/tracks/{filename}.json'
    output_video_path = f'../data/annotations/tracked_videos/{filename}.mp4'
    # output_video_path = f'../data/annotations/labeled_videos/{filename}.mp4'
    output_json_path = f'../data/annotations/labeled_tracks/{filename}.json'
    # create_video_from_json(original_video_path, json_path, output_video_path)
    # if add_audio_to_video(input_path, output_path):
    #     print(f"Audio successfully added to {output_path}")
    # else:
    #     print(f"Failed to add audio to {output_path}")

    # Verify audio streams
    # if check_audio_stream(original_video_path):
    #     print(f"{os.path.basename(original_video_path)}: Audio stream present.")
    # else:
    #     print(f"{os.path.basename(original_video_path)}: No audio stream detected.")
    annotate_speaker_activity(original_video_path, json_path, output_json_path)
    
if __name__ == "__main__":
    main()
