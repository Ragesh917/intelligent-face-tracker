class FaceTracker:
    def __init__(self, max_missing_frames=30, iou_threshold=0.3):
        self.max_missing_frames = max_missing_frames
        self.iou_threshold = iou_threshold

        self.next_track_id = 1
        self.tracks = {}

    def _calculate_iou(self, box_a, box_b):
        """
        Calculate Intersection over Union (IoU)
        between two bounding boxes.
        """

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1
        )

        intersection_area = (
            intersection_width * intersection_height
        )

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union_area = area_a + area_b - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def update(self, detections):
        """
        Update tracker using current frame detections.

        Each detection should contain:
            {
                "bbox": [x1, y1, x2, y2]
            }

        Returns:
            Detections with assigned track IDs.
        """

        results = []

        matched_tracks = set()

        for detection in detections:

            bbox = detection["bbox"]

            best_track_id = None
            best_iou = 0.0

            for track_id, track in self.tracks.items():

                if track_id in matched_tracks:
                    continue

                iou = self._calculate_iou(
                    bbox,
                    track["bbox"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if (
                best_track_id is not None
                and best_iou >= self.iou_threshold
            ):
                track_id = best_track_id

                self.tracks[track_id]["bbox"] = bbox
                self.tracks[track_id]["missing"] = 0

                matched_tracks.add(track_id)

            else:
                track_id = self.next_track_id
                self.next_track_id += 1

                self.tracks[track_id] = {
                    "bbox": bbox,
                    "missing": 0,
                }

                matched_tracks.add(track_id)

            result = dict(detection)
            result["track_id"] = track_id

            results.append(result)

        # Increase missing count for tracks
        # that were not detected in this frame.
        for track_id in list(self.tracks.keys()):

            if track_id not in matched_tracks:

                self.tracks[track_id]["missing"] += 1

                if (
                    self.tracks[track_id]["missing"]
                    > self.max_missing_frames
                ):
                    del self.tracks[track_id]

        return results

    def get_active_tracks(self):
        """
        Return currently active tracks.
        """

        return self.tracks