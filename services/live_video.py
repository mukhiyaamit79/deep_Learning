"""Thread-safe bridge between the WebRTC video thread and the Streamlit script thread.

streamlit-webrtc calls ``video_frame_callback`` in a background thread, where we must NOT call
Streamlit and must NOT write to SQLite. Therefore:

* the video thread calls :meth:`LiveRecognitionSession.process_frame` (detection + CNN, throttled)
  and gets back an annotated frame;
* the Streamlit page reads :meth:`LiveRecognitionSession.snapshot` from an auto-refreshing
  ``st.fragment`` and does the attendance write itself (auto-mark or teacher confirmation).

All shared state is protected by one lock; inference runs outside the lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from services.face_recognition import (
    FaceRecognizer,
    FrameAnalysis,
    FrameStatus,
    RecognitionStabilizer,
    RecognitionThresholds,
    StabilitySnapshot,
)
from utils.image_utils import (
    COLOR_CANDIDATE,
    COLOR_RECOGNIZED,
    COLOR_REJECTED,
    COLOR_UNCERTAIN,
    FaceBox,
    draw_banner,
    draw_face_box,
)


@dataclass(frozen=True)
class LiveConfig:
    thresholds: RecognitionThresholds
    stability_frames: int
    max_gap_s: float
    inference_interval_s: float
    show_boxes: bool = True
    mirror: bool = True
    #: model label -> name drawn on the video. Only active, consenting, linked students are listed;
    #: everyone else is drawn as "Not enrolled".
    overlay_names: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LiveSnapshot:
    analysis: FrameAnalysis | None
    stability: StabilitySnapshot
    frames_received: int
    frames_analyzed: int
    last_frame_at: float | None
    processing_ms: float | None
    frame_size: tuple[int, int] | None

    def seconds_since_last_frame(self) -> float | None:
        return None if self.last_frame_at is None else time.monotonic() - self.last_frame_at


class LiveRecognitionSession:
    """Per-browser-session live recognition state."""

    def __init__(self, recognizer: FaceRecognizer, config: LiveConfig) -> None:
        self.recognizer = recognizer
        self._lock = threading.Lock()
        self._config = config
        self._stabilizer = RecognitionStabilizer(config.stability_frames, config.max_gap_s)
        self._stability = self._stabilizer.snapshot()
        self._analysis: FrameAnalysis | None = None
        self._frames_received = 0
        self._frames_analyzed = 0
        self._last_frame_at: float | None = None
        self._last_analysis_at = 0.0
        self._processing_ms: float | None = None
        self._frame_size: tuple[int, int] | None = None
        self._busy = False

    # ---- Streamlit thread ----------------------------------------------------------------

    def configure(self, config: LiveConfig) -> None:
        with self._lock:
            self._config = config
            self._stabilizer.configure(config.stability_frames, config.max_gap_s)

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            return LiveSnapshot(self._analysis, self._stability, self._frames_received, self._frames_analyzed,
                                self._last_frame_at, self._processing_ms, self._frame_size)

    def reset_stability(self) -> None:
        with self._lock:
            self._stabilizer.reset()
            self._stability = self._stabilizer.snapshot()

    def reset_all(self) -> None:
        with self._lock:
            self._stabilizer.reset()
            self._stability = self._stabilizer.snapshot()
            self._analysis = None
            self._frames_received = self._frames_analyzed = 0
            self._last_frame_at = None
            self._processing_ms = None

    # ---- WebRTC worker thread --------------------------------------------------------------

    def process_frame(self, image_bgr: np.ndarray) -> np.ndarray:
        """Analyse (throttled) and annotate one frame. Never raises."""
        now = time.monotonic()
        with self._lock:
            self._frames_received += 1
            self._last_frame_at = now
            self._frame_size = (int(image_bgr.shape[1]), int(image_bgr.shape[0]))
            config = self._config
            should_analyze = (not self._busy) and (now - self._last_analysis_at >= config.inference_interval_s)
            if should_analyze:
                self._busy = True
                self._last_analysis_at = now

        if should_analyze:
            started = time.perf_counter()
            analysis = self.recognizer.analyze(image_bgr, config.thresholds)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._busy = False
                self._analysis = analysis
                self._stability = self._stabilizer.update(analysis)
                self._frames_analyzed += 1
                self._processing_ms = elapsed_ms

        with self._lock:
            analysis_now, stability_now = self._analysis, self._stability
        try:
            return render_overlay(image_bgr, analysis_now, stability_now, config)
        except Exception:  # noqa: BLE001 - drawing must never break the stream
            return image_bgr


def _mirror(box: FaceBox, width: int) -> FaceBox:
    return FaceBox(width - box.x - box.w, box.y, box.w, box.h)


def render_overlay(image_bgr: np.ndarray, analysis: FrameAnalysis | None, stability: StabilitySnapshot,
                   config: LiveConfig) -> np.ndarray:
    """Annotated copy of the frame. Mirroring is for display only."""
    frame = cv2.flip(image_bgr, 1) if config.mirror else image_bgr.copy()
    width = frame.shape[1]
    if analysis is None:
        return draw_banner(frame, "Starting recognition...", COLOR_UNCERTAIN)
    names = dict(config.overlay_names)
    boxes = [(_mirror(b, width) if config.mirror else b) for b in analysis.faces]

    if analysis.status is FrameStatus.NO_FACE:
        return draw_banner(frame, "No face detected", COLOR_UNCERTAIN)
    if analysis.status is FrameStatus.MULTIPLE_FACES:
        if config.show_boxes:
            for box in boxes:
                draw_face_box(frame, box, None, COLOR_REJECTED)
        return draw_banner(frame, f"{len(boxes)} faces - one student at a time", COLOR_REJECTED)
    if analysis.status in (FrameStatus.ERROR, FrameStatus.INVALID_IMAGE):
        return draw_banner(frame, "Recognition error - see panel", COLOR_REJECTED)

    box = boxes[0] if boxes else None
    if analysis.status is FrameStatus.UNCERTAIN:
        if box is not None and config.show_boxes:
            draw_face_box(frame, box, "Unknown", COLOR_UNCERTAIN)
        return draw_banner(frame, "Unknown / not confident", COLOR_UNCERTAIN)

    assert analysis.prediction is not None
    label = analysis.prediction.label
    display = names.get(label)
    score = analysis.prediction.confidence
    if stability.is_stable and stability.label == label:
        caption = f"{display} {score:.0%}" if display else "Not enrolled"
        color = COLOR_RECOGNIZED if display else COLOR_REJECTED
        banner = f"Recognised: {display}" if display else "Recognised class is not enrolled"
    else:
        caption = f"{display or 'Not enrolled'} {stability.consecutive}/{stability.required}"
        color = COLOR_CANDIDATE
        banner = "Hold still - verifying identity"
    if box is not None and config.show_boxes:
        draw_face_box(frame, box, caption, color)
    return draw_banner(frame, banner, color)
