import cv2
import multiprocessing
import queue
import time

from app.config.loader import load_config
from app.pipeline.ai_worker import create_ai_process


def draw_results(frame, results):
    for result in results:

        x1, y1, x2, y2 = result["bbox"]

        visitor_id = result["visitor_id"]
        track_id = result["track_id"]

        label = (
            f"{visitor_id} | Track {track_id}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(y1 - 10, 20),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


def wait_for_ai_ready(
    ai_process,
    status_queue,
):
    """
    Wait until the independent AI process
    finishes loading YOLO and InsightFace.
    """

    print()
    print(
        "Waiting for AI process to initialize..."
    )

    while ai_process.is_alive():

        try:

            status = status_queue.get(
                timeout=1
            )

            if status == "READY":

                print(
                    "AI process is READY."
                )

                return True

            if status == "ERROR":

                print(
                    "AI process reported an error."
                )

                return False

        except queue.Empty:

            print(
                "AI still initializing..."
            )

    print(
        "AI process stopped before becoming ready."
    )

    return False


def main():

    multiprocessing.freeze_support()

    config = load_config()

    source = config["input"]["source"]

    skip_frames = config[
        "detection"
    ]["skip_frames"]

    confidence = config[
        "detection"
    ]["confidence"]

    similarity_threshold = config[
        "recognition"
    ]["similarity_threshold"]

    max_missing_frames = config[
        "tracking"
    ]["max_missing_frames"]

    database_path = config[
        "database"
    ]["path"]

    pipeline_config = {
        "confidence": confidence,
        "similarity_threshold": (
            similarity_threshold
        ),
        "database_path": database_path,
        "max_missing_frames": (
            max_missing_frames
        ),
    }

    input_queue = multiprocessing.Queue(
        maxsize=1
    )

    output_queue = multiprocessing.Queue(
        maxsize=2
    )

    status_queue = multiprocessing.Queue(
        maxsize=2
    )

    print()
    print(
        "Starting independent AI process..."
    )

    ai_process = create_ai_process(
        input_queue,
        output_queue,
        status_queue,
        pipeline_config,
    )

    ai_process.start()

    print(
        f"AI process started. "
        f"PID: {ai_process.pid}"
    )

    # --------------------------------------------------
    # WAIT FOR AI INITIALIZATION
    # --------------------------------------------------

    ai_ready = wait_for_ai_ready(
        ai_process,
        status_queue,
    )

    if not ai_ready:

        print(
            "Unable to initialize AI process."
        )

        if ai_process.is_alive():

            ai_process.terminate()
            ai_process.join()

        input_queue.close()
        output_queue.close()
        status_queue.close()

        return

    # --------------------------------------------------
    # OPEN VIDEO ONLY AFTER AI IS READY
    # --------------------------------------------------

    video = cv2.VideoCapture(
        source
    )

    if not video.isOpened():

        print(
            f"Unable to open video source: "
            f"{source}"
        )

        try:

            input_queue.put_nowait(None)

        except queue.Full:

            pass

        ai_process.join(
            timeout=5
        )

        if ai_process.is_alive():

            ai_process.terminate()
            ai_process.join()

        input_queue.close()
        output_queue.close()
        status_queue.close()

        return

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

    frame_interval = 1.0 / fps

    frame_count = 0

    ai_frames_sent = 0
    ai_frames_received = 0

    latest_results = []

    total_ai_time = 0.0

    total_read_time = 0.0
    total_queue_time = 0.0
    total_result_time = 0.0
    total_draw_time = 0.0
    total_display_time = 0.0
    total_sleep_time = 0.0

    queue_full_count = 0

    start_time = time.perf_counter()

    print()
    print(
        "Realtime video processing started."
    )

    print(
        f"Video FPS: {fps:.2f}"
    )

    print(
        f"AI skip frames: {skip_frames}"
    )

    print(
        "AI process already warmed up."
    )

    print(
        "Timing diagnostics enabled."
    )

    print(
        "Press Q to stop."
    )

    print()

    try:

        while True:

            loop_start = (
                time.perf_counter()
            )

            # --------------------------------------------------
            # READ FRAME
            # --------------------------------------------------

            read_start = (
                time.perf_counter()
            )

            success, frame = (
                video.read()
            )

            total_read_time += (
                time.perf_counter()
                - read_start
            )

            if not success:

                print(
                    "Video ended."
                )

                break

            frame_count += 1

            # --------------------------------------------------
            # SEND SELECTED FRAME TO AI
            # --------------------------------------------------

            if (
                frame_count
                % skip_frames
                == 0
            ):

                queue_start = (
                    time.perf_counter()
                )

                try:

                    input_queue.put_nowait(
                        (
                            frame_count,
                            frame.copy(),
                        )
                    )

                    ai_frames_sent += 1

                except queue.Full:

                    queue_full_count += 1

                total_queue_time += (
                    time.perf_counter()
                    - queue_start
                )

            # --------------------------------------------------
            # RECEIVE AI RESULTS
            # --------------------------------------------------

            result_start = (
                time.perf_counter()
            )

            while True:

                try:

                    (
                        result_frame_id,
                        results,
                        processing_time,
                    ) = (
                        output_queue.get_nowait()
                    )

                    latest_results = (
                        results
                    )

                    ai_frames_received += 1

                    total_ai_time += (
                        processing_time
                    )

                except queue.Empty:

                    break

            total_result_time += (
                time.perf_counter()
                - result_start
            )

            # --------------------------------------------------
            # DRAW RESULTS
            # --------------------------------------------------

            draw_start = (
                time.perf_counter()
            )

            draw_results(
                frame,
                latest_results,
            )

            total_draw_time += (
                time.perf_counter()
                - draw_start
            )

            # --------------------------------------------------
            # DISPLAY
            # --------------------------------------------------

            display_start = (
                time.perf_counter()
            )

            cv2.imshow(
                "Intelligent Face Tracker",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            total_display_time += (
                time.perf_counter()
                - display_start
            )

            if key == ord("q"):

                print(
                    "Stopped by user."
                )

                break

            # --------------------------------------------------
            # MAINTAIN VIDEO SPEED
            # --------------------------------------------------

            elapsed_loop = (
                time.perf_counter()
                - loop_start
            )

            remaining = (
                frame_interval
                - elapsed_loop
            )

            if remaining > 0:

                sleep_start = (
                    time.perf_counter()
                )

                time.sleep(
                    remaining
                )

                total_sleep_time += (
                    time.perf_counter()
                    - sleep_start
                )

    finally:

        video.release()

        cv2.destroyAllWindows()

        print()
        print(
            "Stopping AI process..."
        )

        try:

            input_queue.put_nowait(
                None
            )

        except queue.Full:

            try:

                input_queue.get_nowait()

            except queue.Empty:

                pass

            try:

                input_queue.put_nowait(
                    None
                )

            except queue.Full:

                pass

        ai_process.join(
            timeout=5
        )

        if ai_process.is_alive():

            print(
                "AI process did not stop normally."
            )

            ai_process.terminate()

            ai_process.join()

        input_queue.close()
        output_queue.close()
        status_queue.close()

    # --------------------------------------------------
    # FINAL DIAGNOSTICS
    # --------------------------------------------------

    total_time = (
        time.perf_counter()
        - start_time
    )

    actual_fps = (
        frame_count / total_time
        if total_time > 0
        else 0
    )

    average_ai_time = (
        total_ai_time
        / ai_frames_received
        if ai_frames_received > 0
        else 0
    )

    print()

    print(
        "========== REALTIME DIAGNOSTICS =========="
    )

    print(
        f"Total frames read: "
        f"{frame_count}"
    )

    print(
        f"AI frames sent: "
        f"{ai_frames_sent}"
    )

    print(
        f"AI frames received: "
        f"{ai_frames_received}"
    )

    print(
        f"Queue full count: "
        f"{queue_full_count}"
    )

    print(
        f"Average AI processing time: "
        f"{average_ai_time:.3f} sec"
    )

    print()

    print(
        f"Total frame read time: "
        f"{total_read_time:.3f} sec"
    )

    print(
        f"Total queue time: "
        f"{total_queue_time:.3f} sec"
    )

    print(
        f"Total result polling time: "
        f"{total_result_time:.3f} sec"
    )

    print(
        f"Total drawing time: "
        f"{total_draw_time:.3f} sec"
    )

    print(
        f"Total display time: "
        f"{total_display_time:.3f} sec"
    )

    print(
        f"Total sleep time: "
        f"{total_sleep_time:.3f} sec"
    )

    print()

    print(
        f"Total elapsed time: "
        f"{total_time:.3f} sec"
    )

    print(
        f"Actual playback FPS: "
        f"{actual_fps:.2f}"
    )

    print(
        "==========================================="
    )

    print()

    print(
        "Timing diagnostics completed."
    )


if __name__ == "__main__":

    main()