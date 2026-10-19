# Intelligent Face Tracker with Auto-Registration and Visitor Counting

## 1. Project Overview

This project implements an AI-driven face tracking and unique visitor counting system for video streams.

The system:
- Detects faces using YOLO.
- Generates face embeddings using InsightFace.
- Automatically registers previously unseen faces.
- Recognizes previously registered visitors using embedding similarity.
- Tracks detected faces across frames.
- Logs entry events with timestamped face images.
- Stores visitor and event metadata in SQLite.
- Writes critical events to `events.log`.
- Supports configurable video input and detection frame skipping.
- Uses NVIDIA CUDA when available.

## 2. Architecture

```text
Video / RTSP Camera
        |
        v
 OpenCV Frame Capture
        |
        v
 YOLO Face Detection
        |
        v
   Face Tracker
        |
        v
 Face Crop + InsightFace
        |
        v
   Face Embedding
        |
        +----------------------+
        |                      |
        v                      v
 Existing Visitor       New Visitor
        |                      |
        v                      v
 Recognition            Auto Registration
        |                      |
        +----------+-----------+
                   |
                   v
             Event Manager
              /         \
             v           v
        SQLite DB     Image Storage
             |
             v
         events.log
```

## 3. Technology Stack

- Python
- OpenCV
- Ultralytics YOLO
- InsightFace / ArcFace embeddings
- ONNX Runtime GPU
- SQLite
- JSON configuration
- Python logging
- NVIDIA CUDA

## 4. Project Structure

```text
Intelligent Face Tracker/
├── app/
│   ├── main.py
│   ├── scan_video.py
│   ├── config/
│   ├── detection/
│   ├── recognition/
│   ├── tracking/
│   ├── registration/
│   ├── events/
│   ├── database/
│   └── pipeline/
├── config.json
├── requirements.txt
├── events.log
├── data/
│   └── database.db
├── logs/
│   ├── entries/
│   └── exits/
├── models/
├── videos/
├── tests/
└── docs/
    ├── AI_PLANNING.md
    └── ARCHITECTURE.md
```

## 5. Configuration

`config.json` controls the video source, detection confidence, frame skipping, recognition threshold, tracking timeout, database path and storage locations.

Example:

```json
{
    "input": {
        "source": "videos/sample.mp4"
    },
    "detection": {
        "skip_frames": 15,
        "confidence": 0.5
    },
    "recognition": {
        "similarity_threshold": 0.5
    },
    "tracking": {
        "max_missing_frames": 30
    },
    "database": {
        "path": "data/database.db"
    },
    "logging": {
        "event_log": "events.log"
    },
    "storage": {
        "entry_images": "logs/entries",
        "exit_images": "logs/exits"
    }
}
```

## 6. Running the Project

Create and activate a Python virtual environment, install the dependencies, place the YOLO face model in `models/`, and ensure the InsightFace `buffalo_l` model is available under `insightface_models/`.

Run the video scanner:

```powershell
python -m app.scan_video
```

Run database verification:

```powershell
python -m tests.check_database
```

## 7. GPU

The tested development machine uses:

- NVIDIA GeForce RTX 3050 Laptop GPU 6GB
- PyTorch CUDA build
- ONNX Runtime GPU
- CUDA-enabled YOLO inference
- CUDA-enabled InsightFace model sessions

The application automatically selects CUDA when available and falls back to CPU when CUDA is unavailable.

## 8. Clean Final Test Result

A clean end-to-end test was performed after clearing the database.

```text
Video frames:          240
AI frames processed:   16
Total processing time: 39.85 sec
Overall processing FPS: 6.02

Unique visitors:       25
Entry events:          25
Exit events:            0
Total events:           25
```

The first processed frame automatically registered new visitors. Subsequent frames recognized those visitors using embedding similarity.

The zero exit count in this particular sample run is due to the short sample video and the configured missing-frame timeout; the exit-state logic is implemented in the event state manager.

## 9. Logging and Storage

Visitor data is stored in SQLite.

The system stores:
- Visitor ID
- Creation timestamp
- First-seen timestamp
- Last-seen timestamp
- Face embedding

Event data stores:
- Event ID
- Visitor ID
- Event type
- Timestamp
- Image path

Entry and exit images are stored under:

```text
logs/entries/YYYY-MM-DD/
logs/exits/YYYY-MM-DD/
```

Critical events are written to:

```text
events.log
```

## 10. Assumptions

- The development sample is a video file.
- The interview environment can provide an RTSP camera stream.
- Face similarity depends on image quality, lighting, pose and camera angle.
- CUDA is used when supported hardware and compatible runtimes are available.
- The current tracker is a lightweight IoU-based tracker designed for the working prototype.

## 11. Future Improvements

- Replace the lightweight IoU tracker with ByteTrack or DeepSORT.
- Improve time-based exit handling for variable processing rates.
- Add RTSP configuration and reconnect handling.
- Add a live unique visitor counter overlay.
- Improve multi-camera support.
- Add stronger recognition quality controls.
- Add performance metrics and monitoring.

## 12. Hackathon Notes

AI-assisted development was used during implementation. The generated code was tested incrementally, including GPU availability, YOLO detection, InsightFace embedding generation, recognition, database operations, event logging and end-to-end video scanning.

The final implementation prioritizes a modular architecture so individual components can be tested and replaced independently.

This project is a part of a hackathon run by https://katomaran.com
