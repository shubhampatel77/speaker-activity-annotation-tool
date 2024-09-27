# import torch
import cv2
# from retinaface import RetinaFace
import numpy as np
import logging
import subprocess
import os
import shutil
import json
import matplotlib.pyplot as plt

# from goturn.helper.BoundingBox import BoundingBox
# from goturn.network.network import GoturnNetwork
# from goturn.helper.image_io import resize

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_goturn_model(ckpt_path):
    model = GoturnNetwork()
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('_model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    logging.info("Loaded state_dict from .ckpt instead.")
    return model.to(device)

def preprocess(im, mean=None):
    if mean is None:
        mean = np.array([104, 117, 123])
    im = resize(im, (227, 227)) - mean
    im = torch.from_numpy(im.transpose(2, 0, 1)).float().to(device)
    return im.unsqueeze(0)

def normalize_bbox(bbox, width, height):
    return BoundingBox(
        bbox.x1 / width,
        bbox.y1 / height,
        bbox.x2 / width,
        bbox.y2 / height
    )

def denormalize_bbox(bbox, width, height):
    return BoundingBox(
        bbox.x1 * width,
        bbox.y1 * height,
        bbox.x2 * width,
        bbox.y2 * height
    )

def expand_bbox(bbox, expansion_factor=1.3):
    center_x = (bbox.x1 + bbox.x2) / 2
    center_y = (bbox.y1 + bbox.y2) / 2
    width = bbox.x2 - bbox.x1
    height = bbox.y2 - bbox.y1
    new_width = width * expansion_factor
    new_height = height * expansion_factor
    new_x1 = max(0, center_x - new_width / 2)
    new_y1 = max(0, center_y - new_height / 2)
    new_x2 = min(1, center_x + new_width / 2)
    new_y2 = min(1, center_y + new_height / 2)
    return BoundingBox(new_x1, new_y1, new_x2, new_y2)

def compute_iou(box1, box2):
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def check_audio_stream(video_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Handle cases where no audio stream is found
    output = result.stdout.strip()
    if output == '':
        return False
    return int(output) > 0

def add_audio_to_video(original_video_path, annotated_video_path):
    if not check_audio_stream(original_video_path):
        print(f"No audio stream found in {original_video_path}")
        return False
    
    # Create a temporary output file to avoid in-place editing issues
    temp_output_path = annotated_video_path.replace(".mp4", "_temp.mp4")
    
    ffmpeg_command = [
        'ffmpeg',
        '-i', annotated_video_path,
        '-i', original_video_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-y',
        temp_output_path  # Temporary output file
    ]
    
    try:
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"FFmpeg output for {temp_output_path}:")
        print(result.stdout)
        print(result.stderr)
        
        # Verify the output video has an audio stream
        if check_audio_stream(temp_output_path):
            # Replace the original file with the new one
            shutil.move(temp_output_path, annotated_video_path)
            return True
        else:
            print(f"Failed to add audio to {temp_output_path}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error processing {annotated_video_path}: {e}")
        print("FFmpeg error output:")
        print(e.stderr)
        return False
    finally:
        # Clean up the temporary file if it still exists
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            

def analyze_track_duration(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    track_durations = {}
    for frame, tracks in json_data.items():
        for track in tracks:
            track_id = track['track_id']
            if track_id not in track_durations:
                track_durations[track_id] = {'start': int(frame), 'end': int(frame)}
            else:
                track_durations[track_id]['end'] = int(frame)

    durations = [d['end'] - d['start'] + 1 for d in track_durations.values()]

    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50)
    plt.title('Distribution of Track Durations')
    plt.xlabel('Duration (frames)')
    plt.ylabel('Frequency')
    plt.show()

    print(f"Mean duration: {np.mean(durations):.2f} frames")
    print(f"Median duration: {np.median(durations):.2f} frames")
    print(f"25th percentile: {np.percentile(durations, 25):.2f} frames")
    print(f"75th percentile: {np.percentile(durations, 75):.2f} frames")

    return durations

def draw_dotted_rectangle(img, pt1, pt2, color, thickness=1, gap=10):
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Scale gap based on box size
    box_width = x2 - x1
    box_height = y2 - y1
    gap = max(int(min(box_width, box_height) / 20), 1)

    # Draw horizontal dotted lines
    for x in range(x1, x2, gap):
        cv2.line(img, (x, y1), (min(x+gap//2, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x+gap//2, x2), y2), color, thickness)
    
    # Draw vertical dotted lines
    for y in range(y1, y2, gap):
        cv2.line(img, (x1, y), (x1, min(y+gap//2, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y+gap//2, y2)), color, thickness)
        
def remove_short_tracks(json_path, threshold=5):  # Using threshold of 5 emperically
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    track_durations = {}
    for frame, tracks in json_data.items():
        for track in tracks:
            track_id = track['track_id']
            if track_id not in track_durations:
                track_durations[track_id] = {'start': int(frame), 'end': int(frame)}
            else:
                track_durations[track_id]['end'] = int(frame)

    long_tracks = {k: v for k, v in track_durations.items() if v['end'] - v['start'] + 1 > threshold}

    filtered_data = {}
    for frame, tracks in json_data.items():
        filtered_tracks = [track for track in tracks if track['track_id'] in long_tracks]
        if filtered_tracks:
            filtered_data[frame] = filtered_tracks

    output_path = os.path.splitext(json_path)[0] + '-r.json'
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered data saved to {output_path}")