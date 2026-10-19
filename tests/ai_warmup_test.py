import cv2
import time

from app.detection.yolo_detector import YOLOFaceDetector
from app.recognition.face_recognizer import FaceRecognizer
from app.pipeline.face_pipeline import crop_face


video = cv2.VideoCapture("videos/sample.mp4")

success, frame = video.read()

video.release()

if not success:
    raise RuntimeError("Could not read sample video.")


print()
print("Initializing AI models...")
print()


detector = YOLOFaceDetector(
    model_path=r"models\yolov11n-face.pt",
    confidence=0.5,
    device=0,
)

recognizer = FaceRecognizer(
    model_name="buffalo_l",
    model_root=r"insightface_models",
    similarity_threshold=0.5,
    device=0,
)


print()
print("Starting CUDA/model warm-up...")
print()


# -------------------------------------------------
# YOLO warm-up
# -------------------------------------------------

start = time.perf_counter()

detections = detector.detect(frame)

yolo_time = time.perf_counter() - start

print(
    f"YOLO warm-up time: {yolo_time:.2f} sec"
)

print(
    f"Faces detected: {len(detections)}"
)


# -------------------------------------------------
# InsightFace warm-up
# -------------------------------------------------

embedding_count = 0

start = time.perf_counter()

for detection in detections:

    face_crop = crop_face(
        frame,
        detection["bbox"],
    )

    if face_crop is None:
        continue

    embedding = recognizer.get_embedding(
        face_crop
    )

    if embedding is not None:
        embedding_count += 1


insightface_time = (
    time.perf_counter() - start
)


print(
    f"InsightFace warm-up time: "
    f"{insightface_time:.2f} sec"
)

print(
    f"Embeddings generated: "
    f"{embedding_count}"
)


# -------------------------------------------------
# Second run to verify warm performance
# -------------------------------------------------

print()
print("Testing warm performance...")
print()


start = time.perf_counter()

detections = detector.detect(frame)

yolo_warm_time = (
    time.perf_counter() - start
)


start = time.perf_counter()

embedding_count = 0

for detection in detections:

    face_crop = crop_face(
        frame,
        detection["bbox"],
    )

    if face_crop is None:
        continue

    embedding = recognizer.get_embedding(
        face_crop
    )

    if embedding is not None:
        embedding_count += 1


insightface_warm_time = (
    time.perf_counter() - start
)


print(
    f"YOLO warm time: "
    f"{yolo_warm_time:.2f} sec"
)

print(
    f"InsightFace warm time: "
    f"{insightface_warm_time:.2f} sec"
)

print(
    f"Embeddings: "
    f"{embedding_count}"
)

print()
print("AI warm-up test completed.")
