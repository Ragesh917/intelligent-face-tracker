import torch
from ultralytics import YOLO


class YOLOFaceDetector:
    def __init__(
        self,
        model_path=r"models\yolov11n-face.pt",
        confidence=0.5,
        device=0,
    ):
        self.device = device

        if torch.cuda.is_available():
            self.device = 0
            print("YOLO device: CUDA")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("YOLO device: CPU")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.confidence = confidence

    def detect(self, frame):
        """
        Detect faces in a single OpenCV frame.

        Returns:
            List of detections:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float
            }
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])

                detections.append(
                    {
                        "bbox": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ],
                        "confidence": confidence,
                    }
                )

        return detections