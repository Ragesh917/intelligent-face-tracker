import cv2
import time

from app.pipeline.face_pipeline import FacePipeline


video = cv2.VideoCapture("videos/sample.mp4")

success, frame = video.read()

video.release()

if not success:
    raise RuntimeError("Could not read sample video.")


pipeline = FacePipeline(
    confidence=0.5,
    similarity_threshold=0.5,
    database_path="data/database.db",
    max_missing_frames=30,
)


print()
print("Testing AI pipeline in MAIN thread...")
print()


start = time.perf_counter()

result1 = pipeline.process_frame(frame)

time1 = time.perf_counter() - start

print(
    f"Main thread AI time: {time1:.2f} sec"
)

print(
    f"Results: {len(result1)}"
)


start = time.perf_counter()

result2 = pipeline.process_frame(frame)

time2 = time.perf_counter() - start

print(
    f"Main thread second AI time: {time2:.2f} sec"
)

print(
    f"Results: {len(result2)}"
)


pipeline.close()

print()
print("Worker-thread problem isolation complete.")
