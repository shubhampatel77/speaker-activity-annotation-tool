import cv2
import json
import os
import numpy as np
import av
import pygame
from threading import Thread
import time

COLORS = {
    'highlighted': (0, 255, 255),  # Yellow
    'non_annotated': (255, 0, 0),  # Blue
    'speaking': (0, 0, 255),  # Red
    'not_speaking': (0, 255, 0),  # Green
    'no_activity': (255, 0, 0)  # Blue (same as non_annotated)
}

def annotate_speaker_activity(input_video_path, tracks_json_path, output_json_path):
    with open(tracks_json_path, 'r') as f:
        json_data = json.load(f)

    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Extract audio using PyAV
    container = av.open(input_video_path)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

    if audio_stream:
        # Extract audio and save as WAV
        temp_audio_path = 'temp_audio.wav'
        out_container = av.open(temp_audio_path, mode='w')
        out_stream = out_container.add_stream('pcm_s16le', rate=audio_stream.rate, layout='stereo')

        for frame in container.decode(audio=0):
            for packet in out_stream.encode(frame):
                out_container.mux(packet)

        # Flush the stream
        for packet in out_stream.encode(None):
            out_container.mux(packet)

        out_container.close()

        # Initialize pygame for audio playback
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.mixer.music.load(temp_audio_path)
    else:
        print("No audio stream found in the video.")

    current_frame = 0
    selected_track_id = None
    track_activities = {}

    cv2.namedWindow('Video')
    cv2.moveWindow('Video', 40, 30)

    is_playing = False
    last_frame_time = 0
    
    # Pre-load frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    def update_display():
        nonlocal current_frame, last_frame_time

        if is_playing:
            current_time = time.time()
            elapsed_time = current_time - last_frame_time
            frame_duration = 1/fps
            if elapsed_time < frame_duration:
                time.sleep(frame_duration - elapsed_time)
            last_frame_time = time.time()
            current_frame += 1
            if current_frame >= total_frames:
                current_frame = 0
                if audio_stream:
                    pygame.mixer.music.play()

        frame = frames[current_frame].copy()
        frame_data = json_data.get(str(current_frame), [])

        # Draw bounding boxes and other information
        for track in frame_data:
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1 = int(bbox[0] * width), int(bbox[1] * height)
            x2, y2 = int(bbox[2] * width), int(bbox[3] * height)

            if track_id in track_activities:
                current_segment = next((seg for seg in track_activities[track_id] 
                                        if seg['start'] <= current_frame <= seg['end']), None)
                if current_segment:
                    color = COLORS[current_segment['activity']]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    color = COLORS['no_activity']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                color = COLORS['non_annotated']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Highlight the selected track with a yellow outline
            if track_id == selected_track_id:
                overlay = frame.copy()
                fill_color = (173, 216, 230)  # Light blue (BGR format)q
                cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
                alpha = 0.5  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            label = f"{track_id}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            label_y = y1 - 10 if y1 - label_height - 10 >= 0 else y1 + label_height + 10
            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            activity = "TODO"
            if track_id in track_activities:
                for segment in track_activities[track_id]:
                    if segment['start'] <= current_frame <= segment['end']:
                        activity = segment['activity']
                        break
            activity_label = {'speaking': 'S', 'not_speaking': 'NS'}.get(activity, 'TODO')
            cv2.putText(frame, activity_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f'Frame: {current_frame}/{total_frames-1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'A/D: Prev/Next frame, W/S: +/-10 frames', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, 'T: Select Track, Z/X/C/V: Set Activity', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, 'Q: Quit', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        return True
    
    # Separate thread for audio playback
    def audio_playback():
        nonlocal current_frame
        while is_playing:
            pygame.mixer.music.set_pos(current_frame / fps)
            time.sleep(0.1)

    audio_thread = None

    while update_display():
        if is_playing:
            key = cv2.waitKey(1) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar to toggle play/pause
            is_playing = not is_playing
            last_frame_time = time.time()
            if is_playing:
                if audio_stream:
                    pygame.mixer.music.play(-1, start=(current_frame / fps))
            else:
                if audio_stream:
                    pygame.mixer.music.pause()
        elif key in [ord('a'), ord('d'), ord('w'), ord('s')]:
            is_playing = False
            if audio_stream:
                pygame.mixer.music.pause()
            if key == ord('a'):
                current_frame = max(current_frame - 1, 0)
            elif key == ord('d'):
                current_frame = min(current_frame + 1, total_frames - 1)
            elif key == ord('w'):
                current_frame = min(current_frame + 10, total_frames - 1)
            elif key == ord('s'):
                current_frame = max(current_frame - 10, 0)
        elif key == ord('t'):
            frame_data = json_data.get(str(current_frame), [])
            track_ids = [track['track_id'] for track in frame_data]
            if track_ids:
                if selected_track_id in track_ids:
                    selected_track_id = track_ids[(track_ids.index(selected_track_id) + 1) % len(track_ids)]
                else:
                    selected_track_id = track_ids[0]
        elif key in [ord('z'), ord('x'), ord('c'), ord('v')] and selected_track_id:
            activity = {ord('z'): "speaking", ord('x'): "speaking", 
                        ord('c'): "not_speaking", ord('v'): "not_speaking"}[key]
            if selected_track_id not in track_activities:
                track_activities[selected_track_id] = []
            
            if key in [ord('z'), ord('c')]:  # Start of activity
                track_activities[selected_track_id].append({'start': current_frame, 'end': current_frame, 'activity': activity})
            elif key in [ord('x'), ord('v')]:  # End of activity
                if track_activities[selected_track_id] and track_activities[selected_track_id][-1]['activity'] == activity:
                    track_activities[selected_track_id][-1]['end'] = current_frame

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    
    # Clean up temporary audio file
    if audio_stream and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    # Update JSON data with activity annotations and fill gaps with "no activity"
    for frame, frame_data in json_data.items():
        frame_num = int(frame)
        for track in frame_data:
            track_id = track['track_id']
            if track_id in track_activities:
                sorted_segments = sorted(track_activities[track_id], key=lambda x: x['start'])
                new_segments = []
                last_end = 0
                for segment in sorted_segments:
                    if segment['start'] > last_end:
                        new_segments.append({'start': last_end, 'end': segment['start'] - 1, 'activity': "no_activity"})
                    new_segments.append(segment)
                    last_end = segment['end'] + 1
                if last_end <= frame_num:
                    new_segments.append({'start': last_end, 'end': frame_num, 'activity': "no_activity"})
                track['segments'] = new_segments
            else:
                track['segments'] = [{'start': frame_num, 'end': frame_num, 'activity': "no_activity"}]

    # Save updated JSON
    with open(output_json_path, 'w') as f:
        json.dump(output_json_path, f, indent=4)

    print("Annotation completed and saved.")