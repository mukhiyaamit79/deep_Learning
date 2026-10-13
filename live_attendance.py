"""Local opt-in attendance demonstration. Run explicitly after notebook NN-36."""
from attendance_runtime import configure_runtime
configure_runtime()

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
from PIL import Image

from attendance_utils import PROJECT_ROOT, load_bundle, predict_face

CSV_COLUMNS = ["Student ID", "Date", "Time", "Confidence", "Status"]


def enroll_for_demo(names):
    """Obtain opt-in for one student; collection/training consent is separate."""
    print("\nOptional academic demo: face processing stays local; no frames are saved.")
    print("This is not liveness detection. A photo or replay may fool recognition.")
    print("Use manual attendance if you do not consent. Type Q to leave.")
    while True:
        student = input("Student: enter your enrolled ID to opt in (Q to quit): ").strip()
        if student.upper() == "Q":
            return None
        if student not in names:
            print("ID is not enrolled in this model. Use manual attendance or try again.")
            continue
        consent = input(f"{student}: consent to face prediction and a confirmed local attendance record? [yes/no]: ")
        if consent.strip().lower() == "yes":
            return student
        print("No consent recorded. No attendance will be written.")


def mark_attendance(directory, student, confidence, now):
    """Append once per student/local date; caller must already have confirmation.

    Run only one demo process per attendance directory. The main loop holds an
    exclusive lock file to prevent concurrent read/check/append races.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    date = now.date().isoformat()
    path = directory / f"attendance_{date}.csv"
    has_content = path.exists() and path.stat().st_size > 0
    if has_content:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != CSV_COLUMNS:
                raise ValueError(f"Unexpected CSV columns in {path}; repair the file before recording.")
            for row in reader:
                if row["Student ID"] == student and row["Date"] == date:
                    return False, path
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if not has_content:
            writer.writeheader()
        writer.writerow({"Student ID": student, "Date": date,
                         "Time": now.strftime("%H:%M:%S%z"),
                         "Confidence": f"{confidence:.6f}", "Status": "Present (confirmed)"})
    return True, path


def box_iou(first, second):
    """Small spatial continuity check; not a biometric tracker or liveness check."""
    if first is None or second is None:
        return 0.0
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(
        0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return intersection / max(1, w1 * h1 + w2 * h2 - intersection)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--attendance-dir", type=Path, default=PROJECT_ROOT / "attendance")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", choices=["cuda"], default="cuda",
                        help="CUDA is required; no CPU fallback is supported.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override heuristic softmax threshold in (0, 1].")
    parser.add_argument("--consistent-frames", type=int, default=None)
    parser.add_argument("--timezone", default="Asia/Katmandu")
    return parser.parse_args()


def main():
    args = parse_args()
    camera = None
    lock_handle = None
    lock_path = None
    try:
        model, names, config = load_bundle(args.model_dir)
        print(f"PyTorch inference device: {next(model.parameters()).device}")
        threshold = config["confidence_threshold"] if args.threshold is None else args.threshold
        required_frames = (int(config["consistent_frames"]) if args.consistent_frames is None
                           else args.consistent_frames)
        if not 0 < threshold <= 1 or required_frames < 2:
            raise ValueError("Threshold must be in (0,1]; consistent-frames must be at least 2.")
        local_timezone = ZoneInfo(args.timezone)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            raise FileNotFoundError(f"Cannot load Haar face detector: {cascade_path}")
        opted_id = enroll_for_demo(names)
        if opted_id is None:
            return 0
        args.attendance_dir.mkdir(parents=True, exist_ok=True)
        lock_path = args.attendance_dir / ".attendance.lock"
        try:
            lock_handle = lock_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise RuntimeError("Another demo may be running. Close it first. If a previous process crashed, "
                               "remove attendance/.attendance.lock manually after verifying no demo is running.") from exc
        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            raise RuntimeError("Cannot open the local webcam. Check permissions, camera index and other camera apps.")
        window = "TECH 405 - Opt-in attendance | Q quit | N next student"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        last_id, previous_box, streak, last_seen = None, None, 0, 0.0
        notice, notice_until = "", 0.0
        print("Preview controls: Q quit; N next student; C confirm when the matching ID is stable.")
        print("Only one face may be present. Confirmation freezes the preview and prompts in this terminal.")
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped providing frames. Attendance was not automatically written.")
            # OpenCV grayscale is for detection only. Classifier preprocessing uses Pillow.
            detection_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(detection_gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(60, 60))
            ready = False
            accepted_score = None
            current_time = time.monotonic()
            if len(faces) != 1:
                last_id, previous_box, streak = None, None, 0
                message = "No face detected" if len(faces) == 0 else "Multiple faces - recording disabled"
                for x, y, w, h in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            else:
                x, y, w, h = (int(value) for value in faces[0])
                current_box = (x, y, w, h)
                crop = Image.fromarray(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
                result = predict_face(model, names, config, crop, threshold)
                student = result["student_id"]
                matches_consent = result["accepted"] and result["candidate"] == opted_id
                continuous = (student == last_id and box_iou(previous_box, current_box) >= 0.25
                              and current_time - last_seen <= 2.0)
                if matches_consent:
                    streak = streak + 1 if continuous else 1
                    last_id, previous_box = student, current_box
                else:
                    last_id, previous_box, streak = None, None, 0
                last_seen = current_time
                ready = matches_consent and streak >= required_frames
                accepted_score = result["confidence"]
                color = (0, 200, 0) if ready else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{student} | score {result['confidence']:.3f}"
                cv2.putText(frame, label, (x, max(22, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if ready:
                    message = f"ID {opted_id} stable: {streak}/{required_frames}. Student: press C to confirm."
                elif result["accepted"] and not matches_consent:
                    message = "Prediction differs from opted-in ID - no record"
                else:
                    message = f"Awaiting stable matching ID: {streak}/{required_frames}"
            cv2.putText(frame, message, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, "Q quit | N next | C confirm | Scores are not calibrated probabilities",
                        (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if current_time < notice_until:
                cv2.putText(frame, notice, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("n"), ord("N")):
                last_id, previous_box, streak = None, None, 0
                opted_id = enroll_for_demo(names)
                if opted_id is None:
                    break
            elif key in (ord("c"), ord("C")) and ready:
                # Frozen preview is a proposal, not an automatic write. Confirmation expires.
                proposed_at = datetime.now(local_timezone)
                confirmation_started = time.monotonic()
                phrase = f"CONFIRM {opted_id}"
                typed = input(f"Student: verify the frozen face/ID. Type '{phrase}' to mark attendance, "
                              "or anything else to cancel (30 second limit): ").strip()
                now = datetime.now(local_timezone)
                if (typed == phrase and time.monotonic() - confirmation_started <= 30
                        and now.date() == proposed_at.date()):
                    marked, path = mark_attendance(args.attendance_dir, opted_id, accepted_score, now)
                    notice = f"Attendance marked: {opted_id}" if marked else f"Already marked today: {opted_id}"
                    print(notice, "->", path)
                else:
                    notice = "Confirmation cancelled or expired - no record written"
                    print(notice)
                notice_until = time.monotonic() + 5
                last_id, previous_box, streak = None, None, 0
    except (OSError, ValueError, RuntimeError, cv2.error, ZoneInfoNotFoundError) as exc:
        print(f"Attendance demo stopped: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nDemo cancelled.")
        return 0
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        if lock_handle is not None:
            lock_handle.close()
            lock_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
