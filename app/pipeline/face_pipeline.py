from app.detection.yolo_detector import YOLOFaceDetector
from app.recognition.face_recognizer import FaceRecognizer
from app.registration.face_registry import FaceRegistry
from app.tracking.tracker import FaceTracker

from app.events.event_manager import EventManager
from app.events.event_state import EventStateManager
from app.events.logger import EventLogger

from app.database.db import get_connection
from app.database.models import create_tables

import cv2


def crop_face(frame, bbox, padding=0.25):
    """
    Crop a detected face with extra padding.
    """

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = map(int, bbox)

    face_width = x2 - x1
    face_height = y2 - y1

    pad_x = int(face_width * padding)
    pad_y = int(face_height * padding)

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return face


class FacePipeline:

    def __init__(
        self,
        detector_model=r"models\yolov11n-face.pt",
        confidence=0.5,
        similarity_threshold=0.5,
        database_path="data/database.db",
        insightface_root=r"insightface_models",
        max_missing_frames=30,
        iou_threshold=0.3,
    ):
        print("Initializing Face Pipeline...")

        self.detector = YOLOFaceDetector(
            model_path=detector_model,
            confidence=confidence,
            device=0,
        )

        self.recognizer = FaceRecognizer(
            model_name="buffalo_l",
            model_root=insightface_root,
            similarity_threshold=similarity_threshold,
            device=0,
        )

        self.connection = get_connection(database_path)
        create_tables(self.connection)

        self.registry = FaceRegistry(
            self.connection,
            similarity_threshold=similarity_threshold,
        )

        self.tracker = FaceTracker(
            max_missing_frames=max_missing_frames,
            iou_threshold=iou_threshold,
        )

        self.event_manager = EventManager(
            self.connection,
            entry_image_dir="logs/entries",
            exit_image_dir="logs/exits",
        )

        self.event_state = EventStateManager(
            max_missing_frames=max_missing_frames,
        )

        self.event_logger = EventLogger(
            log_file="events.log"
        )

        print("Face Pipeline initialized successfully.")

    def process_frame(self, frame):
        """
        Process one video frame.

        YOLO
          ↓
        Tracker
          ↓
        Face embedding
          ↓
        Recognition / registration
          ↓
        Entry / exit state
          ↓
        Database + images + events.log
        """

        detections = self.detector.detect(frame)

        tracked_detections = self.tracker.update(
            detections
        )

        results = []

        active_visitor_ids = set()

        for detection in tracked_detections:

            bbox = detection["bbox"]
            confidence = detection["confidence"]
            track_id = detection["track_id"]

            face_crop = crop_face(
                frame,
                bbox,
            )

            if face_crop is None:
                continue

            embedding = self.recognizer.get_embedding(
                face_crop
            )

            if embedding is None:
                continue

            visitor_id, is_new, similarity = (
                self.registry.recognize_or_register(
                    embedding
                )
            )

            self.registry.update_last_seen(
                visitor_id
            )

            active_visitor_ids.add(
                visitor_id
            )

            # Log recognition/registration.
            if is_new:
                self.event_logger.registration(
                    visitor_id
                )
            else:
                self.event_logger.recognition(
                    visitor_id,
                    similarity,
                )

            self.event_logger.embedding_generated(
                visitor_id
            )

            self.event_logger.tracking(
                visitor_id,
                track_id,
            )

            # Update visitor state.
            state = self.event_state.update(
                visitor_id,
                track_id,
            )

            # First appearance = entry.
            if state == "entry":

                event = self.event_manager.log_event(
                    visitor_id,
                    "entry",
                    face_crop,
                )

                self.event_logger.entry(
                    visitor_id,
                    event["image_path"],
                )

                self.event_state.visitors[
                    visitor_id
                ]["entry_logged"] = True

            results.append(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                    "track_id": track_id,
                    "visitor_id": visitor_id,
                    "is_new": is_new,
                    "similarity": similarity,
                    "state": state,
                }
            )

        # Check for visitors that disappeared.
        exit_visitors = self.event_state.mark_missing(
            active_visitor_ids
        )

        for visitor_id in exit_visitors:

            # We do not have a current face crop because
            # the visitor is no longer visible.
            # The full frame is saved as the exit evidence.
            event = self.event_manager.log_event(
                visitor_id,
                "exit",
                frame,
            )

            self.event_logger.exit(
                visitor_id,
                event["image_path"],
            )

        return results

    def close(self):
        """Close the database connection."""

        if self.connection:
            self.connection.close()