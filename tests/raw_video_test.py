import cv2
import time

video = cv2.VideoCapture("videos/sample.mp4")

fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

print(f"Video FPS: {fps:.2f}")

frame_interval = 1.0 / fps

frames = 0
start = time.perf_counter()

while True:
    loop_start = time.perf_counter()

    success, frame = video.read()

    if not success:
        break

    frames += 1

    cv2.imshow("Raw Video Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    elapsed = time.perf_counter() - loop_start
    remaining = frame_interval - elapsed

    if remaining > 0:
        time.sleep(remaining)

video.release()
cv2.destroyAllWindows()

elapsed = time.perf_counter() - start

print(f"Frames: {frames}")
print(f"Time: {elapsed:.2f} sec")
print(f"Actual FPS: {frames / elapsed:.2f}")
