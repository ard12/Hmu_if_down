"""Main entry point for Fall Detection System."""

import argparse
from pathlib import Path
import sys
import time
import cv2

from src.alert.alert_manager import AlertManager
from src.detector.fall_logic import FallDetector
from src.detector.pose_estimator import PoseEstimator
from src.ui.visualizer import Visualizer
from src.utils import get_project_root, load_config
from src.video.stream_handler import StreamHandler


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Fall Detection System")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. '0'), video file path, or RTSP URL",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable audible alerts",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="File path to save the processed/annotated video (e.g. output/demo_result.mp4)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying a graphical window (useful for servers and tests)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames before terminating",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("      REAL-TIME COMPUTER VISION FALL DETECTION SYSTEM      ")
    print("=" * 60)

    # 1. Load configuration
    try:
        config = load_config(args.config)
        print(f"[Config] Loaded settings from: {args.config}")
    except Exception as e:
        print(f"[Config Error] {e}", file=sys.stderr)
        sys.exit(1)

    # Extract configs
    det_cfg = config.get("detector", {})
    fall_cfg = config.get("fall_heuristics", {})
    alert_cfg = config.get("alert", {})
    vid_cfg = config.get("video", {})

    target_w = vid_cfg.get("frame_width", 960)
    target_h = vid_cfg.get("frame_height", 540)

    # 2. Initialize Video Stream
    print(f"[Stream] Initializing video source: {args.source}...")
    try:
        stream = StreamHandler(
            source=args.source,
            target_width=target_w,
            target_height=target_h,
        )
    except Exception as e:
        print(f"[Stream Error] Failed to open video source: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Initialize Pose Estimator
    print("[Detector] Loading pose estimation model...")
    try:
        estimator = PoseEstimator(
            min_detection_confidence=det_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=det_cfg.get("min_tracking_confidence", 0.5),
            model_complexity=det_cfg.get("model_complexity", 1),
        )
    except Exception as e:
        print(f"[Model Error] {e}", file=sys.stderr)
        stream.release()
        sys.exit(1)

    # 4. Initialize Fall Detector
    fall_detector = FallDetector(
        torso_angle_threshold=fall_cfg.get("torso_angle_threshold", 55.0),
        aspect_ratio_threshold=fall_cfg.get("aspect_ratio_threshold", 0.85),
        fall_velocity_threshold=fall_cfg.get("fall_velocity_threshold", 0.04),
        confirmation_frames=fall_cfg.get("confirmation_frames", 10),
        recovery_frames=fall_cfg.get("recovery_frames", 15),
    )

    # 5. Initialize Alert Manager
    alert_manager = AlertManager(
        enable_sound=alert_cfg.get("enable_sound", True) and not args.no_sound,
        beep_frequency=alert_cfg.get("beep_frequency", 1000),
        beep_duration_ms=alert_cfg.get("beep_duration_ms", 500),
        cooldown_seconds=alert_cfg.get("cooldown_seconds", 5.0),
        enable_logging=alert_cfg.get("enable_logging", True),
        log_dir=alert_cfg.get("log_dir", "incidents"),
    )

    # 6. Initialize Visualizer
    visualizer = Visualizer(draw_skeleton=True)

    # 7. Optional Video Writer
    writer = None
    if args.save_output:
        save_path = Path(args.save_output)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, stream.fps, (target_w, target_h))
        print(f"[Writer] Saving annotated output to: {save_path}")

    print("\n[Running] Press 'q' or 'ESC' in the display window to exit.")
    print("-" * 60)

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = stream.read_frame()
            if not ret or frame is None:
                print("\n[Stream] End of video stream reached.")
                break

            frame_count += 1
            loop_start = time.time()

            # Pose estimation
            landmarks = estimator.process(frame)

            # Evaluate fall kinematics
            metrics = fall_detector.evaluate(landmarks, target_w, target_h)

            # Trigger alerts
            alert_manager.trigger(metrics)

            # Annotate frame
            annotated_frame = visualizer.draw(frame, landmarks, metrics, fps)

            # Save frame if requested
            if writer:
                writer.write(annotated_frame)

            # Calculate FPS
            loop_time = time.time() - loop_start
            fps = 1.0 / max(loop_time, 1e-4)

            # Display window
            if not args.headless:
                cv2.imshow("Fall Detection System", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 27 is ESC
                    print("\n[User] Exit command received.")
                    break

            if args.max_frames and frame_count >= args.max_frames:
                print(f"\n[Limit] Reached max frames ({args.max_frames}).")
                break

    except KeyboardInterrupt:
        print("\n[User] Interrupted by keyboard.")
    finally:
        total_time = time.time() - start_time
        avg_fps = frame_count / max(total_time, 1e-4)
        print(f"\n[Summary] Processed {frame_count} frames in {total_time:.2f}s ({avg_fps:.1f} avg FPS).")

        # Cleanup
        stream.release()
        estimator.close()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[Done] Resources cleaned up successfully.")


if __name__ == "__main__":
    main()
