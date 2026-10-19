import logging
from pathlib import Path


class EventLogger:
    def __init__(self, log_file="events.log"):
        self.log_file = Path(log_file)

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.logger = logging.getLogger(
            "face_tracker"
        )

        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if initialized again.
        if not self.logger.handlers:
            handler = logging.FileHandler(
                self.log_file,
                encoding="utf-8",
            )

            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )

            handler.setFormatter(formatter)

            self.logger.addHandler(handler)

    def log(self, event_type, message):
        """
        Write a critical application event.
        """

        self.logger.info(
            f"{event_type.upper()} | {message}"
        )

    def registration(self, visitor_id):
        self.log(
            "REGISTRATION",
            f"New visitor registered: {visitor_id}",
        )

    def recognition(self, visitor_id, similarity):
        self.log(
            "RECOGNITION",
            f"Visitor recognized: {visitor_id} "
            f"| similarity={similarity:.4f}",
        )

    def tracking(self, visitor_id, track_id):
        self.log(
            "TRACKING",
            f"Visitor tracked: {visitor_id} "
            f"| track_id={track_id}",
        )

    def embedding_generated(self, visitor_id):
        self.log(
            "EMBEDDING_GENERATED",
            f"Embedding generated for: {visitor_id}",
        )

    def entry(self, visitor_id, image_path):
        self.log(
            "ENTRY",
            f"Visitor entered: {visitor_id} "
            f"| image={image_path}",
        )

    def exit(self, visitor_id, image_path):
        self.log(
            "EXIT",
            f"Visitor exited: {visitor_id} "
            f"| image={image_path}",
        )