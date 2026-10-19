# System Architecture

## High-Level Flow

```text
                 +----------------------+
                 | Video File / RTSP    |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | OpenCV Frame Capture |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | YOLO Face Detector   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Face Tracker         |
                 | IoU-based prototype  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Face Crop            |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | InsightFace / ArcFace|
                 | Embedding Generation |
                 +----------+-----------+
                            |
                     +------+------+
                     |             |
                  Match?          New?
                     |             |
                     v             v
              +-----------+  +-----------+
              | Recognize |  | Register  |
              +-----+-----+  +-----+-----+
                    |              |
                    +------+-------+
                           |
                           v
                 +----------------------+
                 | Event State Manager  |
                 +----------+-----------+
                            |
             +--------------+--------------+
             |                             |
             v                             v
     +---------------+             +---------------+
     | SQLite        |             | Image Store   |
     | visitors      |             | entries/exits |
     | events        |             +---------------+
     +-------+-------+
             |
             v
     +---------------+
     | events.log    |
     +---------------+
```

## Module Responsibilities

### Detection
`app/detection/yolo_detector.py`

Detects face bounding boxes using the YOLO face model.

### Recognition
`app/recognition/face_recognizer.py`

Uses InsightFace to generate normalized face embeddings and calculate cosine similarity.

### Registration
`app/registration/face_registry.py`

Searches stored embeddings. If no sufficiently similar visitor exists, a new visitor ID is generated and stored.

### Tracking
`app/tracking/tracker.py`

Maintains lightweight frame-to-frame track IDs using bounding-box IoU.

### Events
`app/events/`

Handles entry/exit state, timestamped images and critical event logging.

### Database
`app/database/`

Creates and manages SQLite persistence for visitors and events.

### Pipeline
`app/pipeline/face_pipeline.py`

Connects detection, recognition, registration, tracking, database and event components into one processing pipeline.

### Scanner
`app/scan_video.py`

Runs the complete pipeline over the development sample video and displays recognition/tracking results.

## Data Model

### Visitors

```text
visitor_id
created_at
first_seen_at
last_seen_at
embedding
```

### Events

```text
event_id
visitor_id
event_type
timestamp
image_path
```

## Failure/Recovery Considerations

- SQLite data is persisted locally.
- Images are saved locally.
- Critical events are written to `events.log`.
- Configuration is externalized into JSON.
- Components are separated so individual failures can be diagnosed.
- The application can be extended with RTSP reconnect handling.

## RTSP Extension

The scanner currently uses a video file for development. For the interview/live deployment, the same OpenCV capture layer can accept an RTSP URL through `config.json`, while the downstream detection, recognition, tracking and storage pipeline remains unchanged.
