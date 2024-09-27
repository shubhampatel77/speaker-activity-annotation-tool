# TODO: create_video_from_json() w/ pretty labels post track annotation, post full annotation
# separate that part from process_videos()
# add audio related functions here

import cv2
import json
import os
import numpy as np
from tqdm import tqdm

from utils import  add_audio_to_video, check_audio_stream

COLORS = {
    'speaking': (0, 0, 255),  # Red
    'not_speaking': (0, 255, 0),  # Green
    'no_activity': (255, 0, 0),  # Blue
    'non_annotated': (255, 0, 0),  # Blue
    'tracking_only': (0, 255, 0)  # Green
}

def draw_label(frame, label, position, color, outside):
    font_scale = 0.7
    thickness = 2
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    if outside:
        cv2.rectangle(frame, (x, y - label_size[1]), (x + label_size[0], y + 5), color, -1)
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - label_size[1]), (x + label_size[0], y + 5), color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    text_color = (255, 255, 255)  # White text for contrast
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def get_label_position(x1, y1, x2, y2, label, font_scale, thickness):
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    label_width, label_height = label_size
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width > label_width * 1.2 and box_height > label_height * 3:
        return (x1 + 5, y1 + label_height + 5), False
    if y1 - label_height - 5 > 0:
        return (x1, y1 - 5), True
    else:
        return (x1, y2 + label_height + 5), True

def create_video_from_json(input_path, json_path, output_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    is_tracking_only = True
    for key, value in json_data.items():
        if value:  # Check if the list is non-empty
            if 'segments' in value[0]:
                is_tracking_only = False
            break  # Stop once you find the first non-empty entry


    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = json_data.get(str(frame_idx), [])
        for track in frame_data:
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1 = int(bbox[0] * width), int(bbox[1] * height)
            x2, y2 = int(bbox[2] * width), int(bbox[3] * height)

            if is_tracking_only:
                activity = "tracking_only"
            else:
                activity = "non_annotated"
                if 'segments' in track:
                    for segment in track['segments']:
                        if segment['start'] <= frame_idx <= segment['end']:
                            activity = segment['activity']
                            break

            color = COLORS[activity]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{track_id}"
            font_scale = 0.7
            thickness = 2
            (label_x, label_y), outside = get_label_position(x1, y1, x2, y2, label, font_scale, thickness)
            draw_label(frame, label, (label_x, label_y), color, outside)

        cv2.putText(frame, f'Frame: {frame_idx}/{total_frames-1}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video creation completed. Output saved to {output_path}")

    print("\nAdding audio back to the annotated video...")

    if add_audio_to_video(input_path, output_path):
        print(f"Audio successfully added to {output_path}")
    else:
        print(f"Failed to add audio to {output_path}")

    # Verify audio streams
    if check_audio_stream(output_path):
        print(f"{os.path.basename(output_path)}: Audio stream present.")
    else:
        print(f"{os.path.basename(output_path)}: No audio stream detected.")
