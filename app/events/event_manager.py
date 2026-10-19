from datetime import datetime, timezone
from pathlib import Path

import cv2


class EventManager:
    def __init__(
        self,
        connection,
        entry_image_dir="logs/entries",
        exit_image_dir="logs/exits",
    ):
        self.connection = connection

        self.entry_image_dir = Path(entry_image_dir)
        self.exit_image_dir = Path(exit_image_dir)

        self.entry_image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.exit_image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _timestamp(self):
        """Return the current UTC timestamp."""

        return datetime.now(timezone.utc)

    def _save_image(
        self,
        frame,
        visitor_id,
        event_type,
        timestamp,
    ):
        """
        Save a timestamped visitor image.
        """

        date_folder = timestamp.strftime("%Y-%m-%d")

        if event_type == "entry":
            base_dir = self.entry_image_dir
        else:
            base_dir = self.exit_image_dir

        folder = base_dir / date_folder
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"{visitor_id}_"
            f"{timestamp.strftime('%H%M%S_%f')}.jpg"
        )

        image_path = folder / filename

        cv2.imwrite(
            str(image_path),
            frame,
        )

        return str(image_path)

    def log_event(
        self,
        visitor_id,
        event_type,
        frame,
    ):
        """
        Log an entry or exit event.

        Returns:
            Dictionary containing event information.
        """

        if event_type not in ("entry", "exit"):
            raise ValueError(
                "event_type must be 'entry' or 'exit'"
            )

        timestamp = self._timestamp()

        image_path = self._save_image(
            frame,
            visitor_id,
            event_type,
            timestamp,
        )

        timestamp_text = timestamp.isoformat()

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO events (
                visitor_id,
                event_type,
                timestamp,
                image_path
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                visitor_id,
                event_type,
                timestamp_text,
                image_path,
            ),
        )

        self.connection.commit()

        return {
            "visitor_id": visitor_id,
            "event_type": event_type,
            "timestamp": timestamp_text,
            "image_path": image_path,
        }