import cv2
import json
import os
import sys
from retinaface import RetinaFace
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import torch
from torchvision import transforms
logging.basicConfig(level=logging.INFO)

from goturn.helper.image_proc import cropPadImage
from goturn.helper.BoundingBox import BoundingBox

from utils import preprocess, normalize_bbox, denormalize_bbox, expand_bbox, compute_iou, add_audio_to_video, check_audio_stream

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_and_process_faces(frame, width, height):
    try:
        detections = RetinaFace.detect_faces(frame)
    except Exception as e:
        logging.error(f"Error in face detection: {str(e)}")
        return []

    current_detections = [
        {
            'bbox':expand_bbox(BoundingBox(
                face['facial_area'][0] / width, 
                face['facial_area'][1] / height,
                face['facial_area'][2] / width, 
                face['facial_area'][3] / height
            )),
            'score': face['score'],
            'landmarks': face['landmarks']
        }
        for face in detections.values()
    ]

    return current_detections

def track_face(curr_frame, prev_frame, prev_bbox, model):
    target_pad, _, _, _ = cropPadImage(prev_bbox, prev_frame)
    cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(prev_bbox, curr_frame)

    target_pad_in = preprocess(target_pad).to(device)
    cur_search_region_in = preprocess(cur_search_region).to(device)

    with torch.no_grad():
        pred_bb = model(target_pad_in, cur_search_region_in)

    pred_bb = BoundingBox(pred_bb[0][0].item(), pred_bb[0][1].item(), pred_bb[0][2].item(), pred_bb[0][3].item())
    pred_bb.unscale(cur_search_region)
    pred_bb.uncenter(curr_frame, search_location, edge_spacing_x, edge_spacing_y)
    return normalize_bbox(pred_bb, curr_frame.shape[1], curr_frame.shape[0])

def match_tracks(tracks, predicted_tracks, current_detections, thresholds):
    matched_tracks = {}
    unmatched_detections = set(range(len(current_detections)))
    
    iou_threshold, fallback_threshold = thresholds

    for track_id, track in tracks.items():
        best_iou = 0
        best_detection_idx = None

        # Step 1: Try matching with GOTURN prediction
        if track_id in predicted_tracks:
            predicted_bbox = predicted_tracks[track_id]
            for i in unmatched_detections:
                iou = compute_iou(predicted_bbox, current_detections[i]['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_detection_idx = i

        # Step 2: If no match, try matching with previous bbox
        if best_iou < iou_threshold:
            prev_bbox = track['bbox']
            for i in unmatched_detections:
                iou = compute_iou(prev_bbox, current_detections[i]['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_detection_idx = i

        if best_detection_idx is not None and best_iou >= fallback_threshold:
            matched_tracks[track_id] = current_detections[best_detection_idx]
            unmatched_detections.remove(best_detection_idx)

    return matched_tracks, unmatched_detections

def process_video(input_video_path, output_json_path, goturn_model, thresholds):
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracks = {}
    next_track_id = 0
    iou_threshold, _, min_track_len = thresholds
    json_data = {}

    try:
        with tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_video_path)}") as pbar:
            ret, prev_frame = cap.read()
            if not ret:
                raise ValueError("Failed to read the first frame")

            # Initialize tracks with first frame detections
            initial_detections = detect_and_process_faces(prev_frame, width, height)
            for detection in initial_detections:
                tracks[str(next_track_id)] = detection
                next_track_id += 1
            # Store initial frame data
            json_data["0"] = [
                {
                    'track_id': track_id,
                    'frame_timestamp': 0,
                    'bbox': [track['bbox'].x1, track['bbox'].y1, track['bbox'].x2, track['bbox'].y2],
                    'predicted_bbox': [0, 0, 0, 0],  # No prediction for the first frame
                    'score': float(track['score']),
                    'landmarks': {k: [float(v[0]), float(v[1])] for k, v in track['landmarks'].items()}
                } for track_id, track in tracks.items()
            ]
            pbar.update(1)
            
            for frame_idx in range(1, total_frames):
                ret, curr_frame = cap.read()
                if not ret:
                    break
                current_detections = detect_and_process_faces(curr_frame, width, height)

                # Predict new positions for existing tracks using GOTURN
                predicted_tracks = {}
                for track_id, track in tracks.items():
                    denormalized_bbox = denormalize_bbox(track['bbox'], width, height)
                    predicted_bbox = track_face(curr_frame, prev_frame, denormalized_bbox, goturn_model)
                    if predicted_bbox is not None:  # Check if GOTURN provided a prediction
                        predicted_tracks[track_id] = predicted_bbox

                # Match tracks with detections
                matched_tracks, unmatched_detections = match_tracks(tracks, predicted_tracks, current_detections, thresholds)

                # Update existing tracks
                tracks = matched_tracks

                # Create new tracks for unmatched detections
                for i in unmatched_detections:
                    track_id = str(next_track_id)
                    next_track_id += 1
                    new_detection = current_detections[i]
                    new_detection['bbox'] = expand_bbox(new_detection['bbox']) 
                    tracks[track_id] = new_detection

                # Save frame data
                frame_data = []
                for track_id, track in tracks.items():
                    bbox = track['bbox']
                    frame_data.append({
                        'track_id': track_id,
                        'frame_timestamp': frame_idx / fps,
                        'bbox': [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                        'predicted_bbox': [float(coord) for coord in expand_bbox(predicted_tracks.get(track_id, BoundingBox(0, 0, 0, 0))).get_bbox()],
                        'score': float(track['score']),
                        'landmarks': {k: [float(v[0]), float(v[1])] for k, v in track['landmarks'].items()}
                    })

                json_data[str(frame_idx)] = frame_data
                prev_frame = curr_frame.copy()
                pbar.update(1)

    except Exception as e:
        logging.error(f"Error processing video {input_path}: {str(e)}")
        raise
    finally:
        cap.release()

    # Remove short tracks
    track_durations = {}
    for frame, tracks in json_data.items():
        for track in tracks:
            track_id = track['track_id']
            if track_id not in track_durations:
                track_durations[track_id] = {'start': int(frame), 'end': int(frame)}
            else:
                track_durations[track_id]['end'] = int(frame)

    long_tracks = {k: v for k, v in track_durations.items() if v['end'] - v['start'] + 1 > min_track_len}

    # Filter and renumber tracks
    filtered_data = {}
    new_track_id = 0
    id_mapping = {}

    for frame, tracks in json_data.items():
        filtered_tracks = []
        for track in tracks:
            if track['track_id'] in long_tracks:
                if track['track_id'] not in id_mapping:
                    id_mapping[track['track_id']] = str(new_track_id)
                    new_track_id += 1
                new_track = track.copy()
                new_track['track_id'] = id_mapping[track['track_id']]
                filtered_tracks.append(new_track)
        if filtered_tracks:
            filtered_data[frame] = filtered_tracks

    # Replace the original json_data with the filtered and renumbered data
    json_data = filtered_data

    # Save the processed JSON data
    with open(output_json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Processed {total_frames} frames")
    print(f"Face tracks data saved to {output_json_path}")