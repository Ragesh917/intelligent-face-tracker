# AI Planning

## Objective

Build an AI-driven unique visitor counter that can detect, recognize, register and track faces from a video stream, while storing entry/exit information locally.

## Development Plan

1. Create modular Python project structure.
2. Load configuration from JSON.
3. Configure GPU-capable PyTorch and ONNX Runtime.
4. Integrate YOLO face detection.
5. Integrate InsightFace embeddings.
6. Create SQLite visitor and event tables.
7. Implement automatic visitor registration.
8. Implement embedding similarity recognition.
9. Implement face tracking.
10. Implement entry/exit state management.
11. Save timestamped event images.
12. Implement mandatory `events.log`.
13. Build an end-to-end video scanner.
14. Test the system using the supplied sample video.
15. Verify database consistency.
16. Prepare README, architecture and demonstration material.

## AI-Assisted Engineering Workflow

AI assistance was used for:
- Architecture planning.
- Module decomposition.
- Code generation.
- Test design.
- Debugging.
- Performance investigation.
- GPU troubleshooting.
- Documentation.

Every major generated component was executed and tested locally.

## Performance Investigation

The system was tested on an NVIDIA GeForce RTX 3050 6GB Laptop GPU.

Cold model initialization and first inference were significantly slower than warm inference. After initialization, YOLO and InsightFace inference became substantially faster.

The final sample-video scan processed:

```text
240 source frames
16 AI processing frames
39.85 seconds total
6.02 overall processing FPS
```

The configurable frame-skip mechanism reduces AI workload while maintaining periodic face recognition.

## Compute Considerations

GPU:
- NVIDIA GeForce RTX 3050 6GB Laptop GPU.

Memory considerations:
- Face detection and recognition models require significant GPU/system memory during initialization.
- ONNX Runtime GPU was used for InsightFace.
- PyTorch CUDA was used for YOLO.

The architecture keeps detection, recognition, tracking, database and event management modular so compute-heavy components can be optimized independently.

## Validation

Validated components include:
- Configuration loading.
- CUDA availability.
- YOLO face detection.
- InsightFace initialization.
- InsightFace embedding generation.
- Visitor registration.
- Visitor recognition.
- Tracking.
- Event state transitions.
- Image event storage.
- SQLite persistence.
- Event logging.
- End-to-end video scanning.
