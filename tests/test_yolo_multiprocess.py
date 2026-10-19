import multiprocessing
import time

import cv2

from app.detection.yolo_detector import YOLOFaceDetector


VIDEO_PATH = r"videos\sample.mp4"


def yolo_worker(result_queue):
    """
    Run YOLO inside an independent process.
    """

    try:

        print(
            "[YOLO PROCESS] Starting...",
            flush=True,
        )

        detector = YOLOFaceDetector(
            model_path=r"models\yolov11n-face.pt",
            confidence=0.5,
            device=0,
        )

        print(
            "[YOLO PROCESS] Model initialized.",
            flush=True,
        )

        video = cv2.VideoCapture(
            VIDEO_PATH
        )

        if not video.isOpened():

            raise RuntimeError(
                f"Unable to open video: {VIDEO_PATH}"
            )

        success, frame = video.read()

        video.release()

        if not success:

            raise RuntimeError(
                "Unable to read first video frame."
            )

        print(
            "[YOLO PROCESS] Frame loaded.",
            flush=True,
        )

        print(
            "[YOLO PROCESS] Starting inference...",
            flush=True,
        )

        start = time.perf_counter()

        detections = detector.detect(
            frame
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            "[YOLO PROCESS] Inference finished.",
            flush=True,
        )

        print(
            f"[YOLO PROCESS] Time: "
            f"{elapsed:.3f} sec",
            flush=True,
        )

        print(
            f"[YOLO PROCESS] Faces: "
            f"{len(detections)}",
            flush=True,
        )

        result_queue.put(
            {
                "success": True,
                "time": elapsed,
                "faces": len(detections),
            }
        )

    except Exception as error:

        print(
            "[YOLO PROCESS] ERROR:",
            error,
            flush=True,
        )

        result_queue.put(
            {
                "success": False,
                "error": str(error),
            }
        )


def main():

    multiprocessing.freeze_support()

    result_queue = multiprocessing.Queue(
        maxsize=1
    )

    print()
    print(
        "=========================================="
    )
    print(
        " YOLO MULTIPROCESS TEST"
    )
    print(
        "=========================================="
    )
    print()

    process = multiprocessing.Process(
        target=yolo_worker,
        args=(
            result_queue,
        ),
    )

    start = time.perf_counter()

    process.start()

    print(
        f"[MAIN] YOLO process started. "
        f"PID: {process.pid}"
    )

    print(
        "[MAIN] Waiting for YOLO process..."
    )

    process.join(
        timeout=30
    )

    total_time = (
        time.perf_counter()
        - start
    )

    if process.is_alive():

        print()
        print(
            "[MAIN] YOLO process DID NOT FINISH "
            "within 30 seconds."
        )

        print(
            "[MAIN] Terminating process..."
        )

        process.terminate()

        process.join()

        print()
        print(
            "RESULT: YOLO MULTIPROCESS HANG"
        )

    else:

        print()
        print(
            "[MAIN] YOLO process finished."
        )

        print(
            f"[MAIN] Total time: "
            f"{total_time:.3f} sec"
        )

        try:

            result = result_queue.get_nowait()

            print()

            if result["success"]:

                print(
                    "RESULT: YOLO MULTIPROCESS "
                    "SUCCESS"
                )

                print(
                    f"YOLO inference time: "
                    f"{result['time']:.3f} sec"
                )

                print(
                    f"Faces detected: "
                    f"{result['faces']}"
                )

            else:

                print(
                    "RESULT: YOLO PROCESS ERROR"
                )

                print(
                    result["error"]
                )

        except Exception:

            print(
                "RESULT: NO RESULT RECEIVED"
            )

    result_queue.close()

    print()
    print(
        "=========================================="
    )
    print()


if __name__ == "__main__":

    main()