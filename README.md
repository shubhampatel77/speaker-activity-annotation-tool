# Active Speaker Detection and Annotation Tool

**This is a comprehensive solution for automatic face detection and tracking and a python-based speaker activity annotation tool for videos.**
## Table of Contents

1. [Active Speaker Detection Annotation Tool](#active-speaker-detection-annotation-tool)
2. [Features](#features)
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Configuration](#configuration)
6. [Output Format](#output-format)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

## Overview
The toolkit consists of several components:
- Face detection and tracking across video frames
- Manual annotation interface for labeling speaking/non-speaking faces
- Visualization tools for displaying annotations
- Utilities for handling video processing and audio synchronization

## Features

- **Robust Face Tracking**: Uses RetinaFace for detection and GOTURN for tracking between frames
- **Interactive Annotation UI**: User-friendly interface for labeling speaker activity
- **Batch Processing**: Process multiple videos in sequence
- **Audio Integration**: Preserves audio during visualization and annotation
- **Customizable Parameters**: Adjustable thresholds for detection and tracking
- **Export Formats**: Save annotations in standard JSON format

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- FFmpeg for audio processing

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/active-speaker-detection-tool.git
cd active-speaker-detection-tool
```

2. Run the setup script to install dependencies:
```bash
chmod +x setup.sh
./setup.sh
```

3. Download the GOTURN pre-trained model:
```bash
# Ensure the goturn-pytorch submodule is initialized
git submodule update --init --recursive
```

## Usage

### Directory Structure

Organize your data in the following structure:
```
├── dataset/
│   ├── original_videos/         # Input videos
│   └── annotations/
│       ├── tracks/              # Face tracking JSON files
│       ├── tracked_videos/      # Videos with face tracking visualization
│       └── labeled_tracks/      # Final speaker activity annotations
```

### Face Tracking

Process videos to detect and track faces:

```bash
python main.py
```

This will:
1. Check all videos in the original_videos directory
2. Process each video to detect and track faces
3. Generate JSON files with tracking data in the annotations/tracks directory

### Speaker Activity Annotation

After tracking, annotate which faces are speaking:

```python
from annotate import annotate_speaker_activity

annotate_speaker_activity(
    "dataset/original_videos/video.mp4",
    "dataset/annotations/tracks/video.json",
    "dataset/annotations/labeled_tracks/video.json"
)
```

The annotation interface provides the following controls:
- **A/D**: Previous/Next frame
- **W/S**: Jump +/-10 frames
- **T**: Select Track
- **Z/X**: Mark as speaking (start/end)
- **C/V**: Mark as not speaking (start/end)
- **Spacebar**: Play/Pause
- **Q**: Quit and save

### Visualization

Generate visualized videos with annotations:

```python
from demonstrate import create_video_from_json

create_video_from_json(
    "dataset/original_videos/video.mp4",
    "dataset/annotations/labeled_tracks/video.json",
    "dataset/annotations/videos/video.mp4"
)
```

## Configuration

Key parameters can be adjusted in `main.py`:

```python
# Detection and tracking thresholds
thresholds = 0.2, 0.2, 5  # IoU threshold, fallback threshold, min track length
```

## Output Format

The annotation tool produces JSON files with the following structure:

```json
{
  "frame_number": [
    {
      "track_id": "0",
      "frame_timestamp": 0.033,
      "bbox": [0.1, 0.2, 0.3, 0.4],
      "score": 0.99,
      "segments": [
        {
          "start": 10,
          "end": 30,
          "activity": "speaking"
        }
      ]
    }
  ]
}
```

## Troubleshooting

- **Video loading errors**: Check if the video codec is supported by OpenCV
- **Face detection issues**: Try adjusting the detection thresholds
- **Tracking problems**: Ensure adequate lighting in videos for better tracking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RetinaFace for face detection
- GOTURN tracker implementation
