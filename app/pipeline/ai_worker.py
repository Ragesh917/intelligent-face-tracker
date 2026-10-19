import multiprocessing
import time

from app.pipeline.face_pipeline import FacePipeline


def ai_worker_process(
    input_queue,
    output_queue,
    status_queue,
    pipeline_config,
):
    """
    Independent AI process.

    This process owns its own FacePipeline.

    Diagnostic timing is enabled around every
    received frame so we can measure the actual
    AI processing time inside the separate process.
    """

    pipeline = None

    try:

        print(
            "[AI PROCESS] Initializing Face Pipeline...",
            flush=True,
        )

        pipeline = FacePipeline(
            confidence=pipeline_config[
                "confidence"
            ],
            similarity_threshold=pipeline_config[
                "similarity_threshold"
            ],
            database_path=pipeline_config[
                "database_path"
            ],
            max_missing_frames=pipeline_config[
                "max_missing_frames"
            ],
        )

        print(
            "[AI PROCESS] Face Pipeline initialized.",
            flush=True,
        )

        status_queue.put(
            "READY"
        )

        print(
            "[AI PROCESS] READY signal sent.",
            flush=True,
        )

        while True:

            item = input_queue.get()

            if item is None:

                print(
                    "[AI PROCESS] Shutdown requested.",
                    flush=True,
                )

                break

            frame_id, frame = item

            print(
                f"[AI PROCESS] FRAME RECEIVED "
                f"| frame={frame_id}",
                flush=True,
            )

            try:

                print(
                    f"[AI PROCESS] AI PROCESSING START "
                    f"| frame={frame_id}",
                    flush=True,
                )

                start = time.perf_counter()

                results = (
                    pipeline.process_frame(
                        frame
                    )
                )

                processing_time = (
                    time.perf_counter()
                    - start
                )

                print(
                    f"[AI PROCESS] AI PROCESSING END "
                    f"| frame={frame_id} "
                    f"| time={processing_time:.3f}s "
                    f"| results={len(results)}",
                    flush=True,
                )

                output_queue.put(
                    (
                        frame_id,
                        results,
                        processing_time,
                    )
                )

                print(
                    f"[AI PROCESS] RESULT SENT "
                    f"| frame={frame_id}",
                    flush=True,
                )

            except Exception as error:

                print(
                    "[AI PROCESS] Processing error:",
                    error,
                    flush=True,
                )

                output_queue.put(
                    (
                        frame_id,
                        [],
                        0.0,
                    )
                )

    except Exception as error:

        print(
            "[AI PROCESS] Initialization error:",
            error,
            flush=True,
        )

        try:

            status_queue.put(
                "ERROR"
            )

        except Exception:

            pass

    finally:

        if pipeline is not None:

            pipeline.close()

        print(
            "[AI PROCESS] Stopped.",
            flush=True,
        )


def create_ai_process(
    input_queue,
    output_queue,
    status_queue,
    pipeline_config,
):
    """
    Create the independent AI process.
    """

    process = multiprocessing.Process(
        target=ai_worker_process,
        args=(
            input_queue,
            output_queue,
            status_queue,
            pipeline_config,
        ),
        daemon=True,
    )

    return process