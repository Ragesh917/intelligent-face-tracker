import cv2
import time

from app.config.loader import load_config
from app.pipeline.face_pipeline import FacePipeline


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


def main():

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

    print()
    print(
        "=========================================="
    )
    print(
        " INTELLIGENT FACE TRACKER - VIDEO SCAN"
    )
    print(
        "=========================================="
    )
    print()

    print(
        "Initializing AI pipeline..."
    )

    pipeline = FacePipeline(
        confidence=confidence,
        similarity_threshold=(
            similarity_threshold
        ),
        database_path=database_path,
        max_missing_frames=(
            max_missing_frames
        ),
    )

    print(
        "AI pipeline ready."
    )

    video = cv2.VideoCapture(
        source
    )

    if not video.isOpened():

        pipeline.close()

        raise RuntimeError(
            f"Unable to open video: {source}"
        )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

    frame_count = 0
    processed_frames = 0

    start_time = time.perf_counter()

    latest_results = []

    print()
    print(
        f"Video FPS: {fps:.2f}"
    )

    print(
        f"AI scan interval: "
        f"every {skip_frames} frames"
    )

    print(
        "Scanning..."
    )

    print(
        "Press Q to stop."
    )

    print()

    try:

        while True:

            success, frame = (
                video.read()
            )

            if not success:

                print(
                    "Video ended."
                )

                break

            frame_count += 1

            if (
                frame_count
                % skip_frames
                == 0
            ):

                process_start = (
                    time.perf_counter()
                )

                latest_results = (
                    pipeline.process_frame(
                        frame
                    )
                )

                processing_time = (
                    time.perf_counter()
                    - process_start
                )

                processed_frames += 1

                print(
                    f"Frame {frame_count:03d} "
                    f"| Faces: "
                    f"{len(latest_results)} "
                    f"| AI time: "
                    f"{processing_time:.2f}s"
                )

                for result in latest_results:

                    print(
                        f"    "
                        f"{result['visitor_id']} "
                        f"| Track "
                        f"{result['track_id']} "
                        f"| New="
                        f"{result['is_new']} "
                        f"| Similarity="
                        f"{result['similarity']:.3f}"
                    )

            draw_results(
                frame,
                latest_results,
            )

            cv2.imshow(
                "Intelligent Face Tracker - Scan",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                print(
                    "Stopped by user."
                )

                break

    finally:

        video.release()

        cv2.destroyAllWindows()

        pipeline.close()

    total_time = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        "=========================================="
    )
    print(
        " SCAN COMPLETE"
    )
    print(
        "=========================================="
    )

    print(
        f"Total frames: "
        f"{frame_count}"
    )

    print(
        f"AI frames processed: "
        f"{processed_frames}"
    )

    print(
        f"Total time: "
        f"{total_time:.2f} sec"
    )

    if total_time > 0:

        print(
            f"Overall processing FPS: "
            f"{frame_count / total_time:.2f}"
        )

    print(
        "=========================================="
    )
    print()


if __name__ == "__main__":

    main()