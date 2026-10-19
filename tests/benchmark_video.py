import cv2
import time


VIDEO_PATH = r"videos\sample.mp4"


def main():

    video = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not video.isOpened():

        raise RuntimeError(
            f"Unable to open video: {VIDEO_PATH}"
        )

    source_fps = video.get(
        cv2.CAP_PROP_FPS
    )

    if source_fps <= 0:

        source_fps = 30.0

    frame_count = 0

    start_time = time.perf_counter()

    print()
    print(
        "========== OPENCV VIDEO BENCHMARK =========="
    )

    print(
        f"Source FPS: {source_fps:.2f}"
    )

    print(
        "Reading and displaying video only."
    )

    print(
        "No YOLO."
    )

    print(
        "No InsightFace."
    )

    print(
        "No multiprocessing."
    )

    print(
        "Press Q to stop."
    )

    print()

    while True:

        success, frame = video.read()

        if not success:

            break

        frame_count += 1

        cv2.imshow(
            "OpenCV Benchmark",
            frame,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break

    elapsed = (
        time.perf_counter()
        - start_time
    )

    video.release()

    cv2.destroyAllWindows()

    actual_fps = (
        frame_count / elapsed
        if elapsed > 0
        else 0
    )

    print()

    print(
        "========== BENCHMARK RESULT =========="
    )

    print(
        f"Frames read: {frame_count}"
    )

    print(
        f"Total time: {elapsed:.3f} sec"
    )

    print(
        f"Actual FPS: {actual_fps:.2f}"
    )

    print(
        "======================================"
    )

    print()


if __name__ == "__main__":

    main()