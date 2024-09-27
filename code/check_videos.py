import cv2
import os

def check_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading first frame from: {video_path}")
        cap.release()
        return False
    
    print(f"Successfully opened and read from: {video_path}")
    print(f"Video properties:")
    print(f"Frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    cap.release()
    return True

def check_videos_in_directory(video_dir):
    successful_loads = []
    failed_loads = []
    
    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):
            full_path = os.path.join(video_dir, filename)
            print(f"\nChecking {filename}:")
            if check_video(full_path):
                successful_loads.append(full_path)
            else:
                failed_loads.append(full_path)
    
    print("\nSummary:")
    print(f"Successfully loaded videos: {len(successful_loads)}")
    print(f"Failed to load videos: {len(failed_loads)}")
    
    return successful_loads, failed_loads
