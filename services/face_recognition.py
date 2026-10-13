"""Face detection, PyTorch face classification and multi-frame stability.

Reuses the notebook's own code so the web app behaves exactly like the trained model expects:
* ``attendance_models.AttendanceCNN`` — the architecture, rebuilt from models/model_metadata.json;
* ``attendance_utils.preprocess_image`` — Pillow grayscale "L", bilinear 128×128, float32 / 255;
* the same validation rules as ``attendance_utils.load_bundle`` for the four model files.

Difference from the notebook: the web app runs inference on CUDA when available and falls back to
the CPU otherwise (the trained CNN takes a few milliseconds per face on a laptop CPU). Training in
the notebook still requires CUDA.

Separate concerns:
1. :class:`FaceDetector` — OpenCV Haar cascade (finds faces, no identity);
2. :class:`FaceClassifier` — the CNN on ONE face crop;
3. :class:`RecognitionStabilizer` — same identity across consecutive frames.
Writing attendance happens elsewhere (services/attendance_service.py).

Limitation: the CNN is a closed-set classifier over the 5 trained students. A stranger is still
assigned to one of them, sometimes with a high score. Threshold + stability reduce, but cannot
eliminate, false matches. No liveness detection: a photo of a student can be recognised.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from attendance_utils import DEFAULT_PREPROCESSING, preprocess_image
from utils.config import AppPaths
from utils.image_utils import (
    FaceBox,
    InvalidImageError,
    bgr_to_pil,
    box_iou,
    crop_box,
    crop_face,
    expand_box,
    shift_box,
    to_grayscale,
    validate_image_array,
)


class ModelArtifactError(RuntimeError):
    """The trained model or its metadata is missing, invalid or inconsistent."""


#: Preprocessing keys that must match attendance_utils.DEFAULT_PREPROCESSING exactly.
CONTRACT_KEYS = ("height", "width", "channels", "color_mode", "resize", "normalization_divisor", "dtype", "input_layout")
REQUIRED_FILES = ("model_path", "class_names_path", "preprocessing_path", "metadata_path")

# ======================================================================================
# Model files (no PyTorch import needed)
# ======================================================================================


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelArtifactError(f"{what} not found: {Path(path).name}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"{what} is not valid JSON: {Path(path).name} ({exc})") from exc


def read_class_names(path: Path) -> tuple[str, ...]:
    """models/class_names.json: ordered list, index i = model output i (the notebook's LabelEncoder)."""
    data = _read_json(path, "Class mapping")
    names = data.get("class_names") if isinstance(data, dict) else data
    if (not isinstance(names, list) or len(names) < 2 or not all(isinstance(n, str) and n for n in names)
            or len(set(names)) != len(names)):
        raise ModelArtifactError("class_names.json must be an ordered list of at least two unique student IDs.")
    return tuple(names)


def read_preprocessing_config(path: Path) -> dict[str, Any]:
    config = _read_json(path, "Preprocessing configuration")
    if not isinstance(config, dict):
        raise ModelArtifactError("preprocessing_config.json must be a JSON object.")
    for key in CONTRACT_KEYS:
        if config.get(key) != DEFAULT_PREPROCESSING[key]:
            raise ModelArtifactError(f"Unsupported preprocessing setting: {key}={config.get(key)!r}.")
    try:
        config["confidence_threshold"] = float(config.get("confidence_threshold", DEFAULT_PREPROCESSING["confidence_threshold"]))
        config["consistent_frames"] = int(config.get("consistent_frames", DEFAULT_PREPROCESSING["consistent_frames"]))
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError("Invalid confidence policy in preprocessing_config.json.") from exc
    return config


def read_metadata(path: Path) -> dict[str, Any]:
    metadata = _read_json(path, "Model metadata")
    if not isinstance(metadata, dict):
        raise ModelArtifactError("model_metadata.json must be a JSON object.")
    if metadata.get("framework") != "pytorch" or metadata.get("format_version") != 1:
        raise ModelArtifactError("model_metadata.json is not a supported PyTorch bundle. Re-run notebook NN-36.")
    if not isinstance(metadata.get("architecture_config"), dict):
        raise ModelArtifactError("model_metadata.json has no architecture_config.")
    return metadata


@dataclass(frozen=True)
class ArtifactStatus:
    model_path: Path
    model_exists: bool
    class_names: tuple[str, ...] | None
    config: dict[str, Any] | None
    metadata: dict[str, Any] | None
    errors: tuple[str, ...]
    fingerprint: tuple[Any, ...]

    @property
    def ready(self) -> bool:
        return self.model_exists and None not in (self.class_names, self.config, self.metadata) and not self.errors

    @property
    def label_set(self) -> set[str] | None:
        return set(self.class_names) if self.class_names is not None else None


def _fingerprint(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return str(path), stat.st_mtime_ns, stat.st_size


def inspect_artifacts(paths: AppPaths) -> ArtifactStatus:
    """Cheap check of the four NN-36 files (reads three small JSON files)."""
    errors: list[str] = []
    names = config = metadata = None
    model_exists = paths.model_path.is_file()
    if not model_exists:
        errors.append(f"Trained model not found: models/{paths.model_path.name} — run notebook NN-36.")
    for loader, path, label in ((read_class_names, paths.class_names_path, "names"),
                                (read_preprocessing_config, paths.preprocessing_path, "config"),
                                (read_metadata, paths.metadata_path, "metadata")):
        try:
            value = loader(path)
        except ModelArtifactError as exc:
            errors.append(str(exc))
            continue
        if label == "names":
            names = value
        elif label == "config":
            config = value
        else:
            metadata = value
    if names is not None and metadata is not None:
        if metadata["architecture_config"].get("num_classes") != len(names):
            errors.append("Model architecture and class mapping do not agree (number of classes).")
    fingerprint = tuple(_fingerprint(getattr(paths, attr)) for attr in REQUIRED_FILES)
    return ArtifactStatus(paths.model_path, model_exists, names, config, metadata, tuple(errors), fingerprint)


# ======================================================================================
# Model loading (PyTorch imported lazily)
# ======================================================================================


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    class_names: tuple[str, ...]
    config: dict[str, Any]
    metadata: dict[str, Any]
    device: str
    torch_version: str
    parameter_count: int
    load_seconds: float


def load_model_bundle(paths: AppPaths, prefer_cuda: bool = True) -> ModelBundle:
    """Rebuild AttendanceCNN from the metadata, load the weights and validate everything."""
    status = inspect_artifacts(paths)
    if not status.ready:
        raise ModelArtifactError("\n".join(status.errors))
    assert status.class_names is not None and status.config is not None and status.metadata is not None
    started = time.perf_counter()
    try:
        import torch

        from attendance_models import AttendanceCNN, count_parameters
    except ImportError as exc:
        raise ModelArtifactError(f"PyTorch is not installed in this Python environment ({exc}).") from exc

    try:
        model = AttendanceCNN(**status.metadata["architecture_config"])
        state = torch.load(paths.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001 - torch raises many types for bad files
        raise ModelArtifactError(f"Could not load {paths.model_path.name}: {exc}") from exc

    device = torch.device("cuda:0" if prefer_cuda and torch.cuda.is_available() else "cpu")
    # Same float32 policy as attendance_utils.load_bundle.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model.to(device).eval()
    height, width = int(status.config["height"]), int(status.config["width"])
    with torch.inference_mode():  # warm-up so the first live frame is not slow
        model(torch.zeros((1, 1, height, width), dtype=torch.float32, device=device))
    return ModelBundle(
        model=model,
        class_names=status.class_names,
        config=status.config,
        metadata=status.metadata,
        device=str(device),
        torch_version=str(torch.__version__),
        parameter_count=int(count_parameters(model)),
        load_seconds=time.perf_counter() - started,
    )


# ======================================================================================
# Typed results
# ======================================================================================


@dataclass(frozen=True)
class RecognitionThresholds:
    confidence_threshold: float = 0.90
    min_margin: float = 0.0
    min_face_size_px: int = 60
    face_crop_margin: float = 0.0


@dataclass(frozen=True)
class Prediction:
    class_index: int
    label: str
    confidence: float  # softmax score of the top class (not a calibrated probability)
    margin: float  # top-1 minus top-2 score
    top_k: tuple[tuple[str, float], ...]

    def rejection_reason(self, t: RecognitionThresholds) -> str | None:
        if self.confidence < t.confidence_threshold:
            return f"score {self.confidence:.0%} is below the {t.confidence_threshold:.0%} threshold"
        if self.margin < t.min_margin:
            return f"lead over the 2nd student ({self.margin:.0%}) is below {t.min_margin:.0%}"
        return None

    def passes(self, t: RecognitionThresholds) -> bool:
        return self.rejection_reason(t) is None


class FrameStatus(str, Enum):
    CANDIDATE = "candidate"  # one face and the score passes (one frame only — not yet stable)
    UNCERTAIN = "uncertain"  # one face, score too low -> "Unknown"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    INVALID_IMAGE = "invalid_image"
    ERROR = "error"


@dataclass(frozen=True)
class FrameAnalysis:
    status: FrameStatus
    faces: tuple[FaceBox, ...]
    prediction: Prediction | None
    message: str
    timestamp: float  # time.monotonic()

    @property
    def candidate_label(self) -> str | None:
        return self.prediction.label if self.status is FrameStatus.CANDIDATE and self.prediction else None

    @property
    def face(self) -> FaceBox | None:
        return self.faces[0] if len(self.faces) == 1 else None


@dataclass(frozen=True)
class StillAnalysis:
    analysis: FrameAnalysis
    variant_predictions: tuple[Prediction, ...]
    agreement: float
    consistent: bool


# ======================================================================================
# 1. Detection
# ======================================================================================


class FaceDetector:
    """OpenCV Haar cascade with the notebook live script's parameters (1.1, 5 neighbours)."""

    def __init__(self, cascade_path: Path | None = None, *, scale_factor: float = 1.1, min_neighbors: int = 5,
                 detection_max_width: int = 640) -> None:
        path = cascade_path or Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise ModelArtifactError(f"OpenCV could not load the face detector from {path}. Reinstall opencv-python.")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.detection_max_width = detection_max_width
        self._lock = threading.Lock()

    def detect(self, gray: np.ndarray, min_face_size_px: int = 60) -> list[FaceBox]:
        """Face boxes in ``gray`` coordinates, largest first. OpenCV grayscale is used for detection only."""
        height, width = gray.shape[:2]
        scale, work = 1.0, gray
        if width > self.detection_max_width:
            scale = self.detection_max_width / float(width)
            work = cv2.resize(gray, (self.detection_max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
        min_size = max(int(round(min_face_size_px * scale)), 20)
        with self._lock:
            rects = self._cascade.detectMultiScale(work, scaleFactor=self.scale_factor,
                                                   minNeighbors=self.min_neighbors, minSize=(min_size, min_size))
        boxes = [FaceBox(int(round(x / scale)), int(round(y / scale)), int(round(w / scale)), int(round(h / scale)))
                 for (x, y, w, h) in (rects if len(rects) else [])]
        return sorted(boxes, key=lambda b: b.area, reverse=True)


# ======================================================================================
# 2. Classification
# ======================================================================================


class FaceClassifier:
    """Runs the CNN on face crops, exactly like attendance_utils.predict_face (softmax over logits)."""

    def __init__(self, bundle: ModelBundle, top_k: int = 3) -> None:
        self.bundle = bundle
        self.top_k = max(1, min(top_k, len(bundle.class_names)))
        self._lock = threading.Lock()  # video thread + UI thread share one model

    def predict(self, faces_rgb: list[Any]) -> list[Prediction]:
        """``faces_rgb``: Pillow images. Preprocessing = attendance_utils.preprocess_image."""
        import torch

        batch = np.stack([preprocess_image(face, self.bundle.config) for face in faces_rgb]).astype(np.float32)
        with self._lock, torch.inference_mode():
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(self.bundle.device)
            scores = torch.softmax(self.bundle.model(tensor), dim=1).cpu().numpy()
        if scores.shape != (len(faces_rgb), len(self.bundle.class_names)) or not np.isfinite(scores).all():
            raise ModelArtifactError("The model returned invalid class scores.")
        return [self._to_prediction(row) for row in scores]

    def _to_prediction(self, scores: np.ndarray) -> Prediction:
        order = np.argsort(scores)[::-1]
        top = int(order[0])
        second = float(scores[order[1]]) if len(order) > 1 else 0.0
        names = self.bundle.class_names
        return Prediction(top, names[top], float(scores[top]), float(scores[top]) - second,
                          tuple((names[int(i)], float(scores[int(i)])) for i in order[: self.top_k]))


class FaceRecognizer:
    """Detection + classification of one image. Holds no per-user state, so it can be cached and shared."""

    def __init__(self, bundle: ModelBundle, detector: FaceDetector | None = None) -> None:
        self.bundle = bundle
        self.detector = detector or FaceDetector()
        self.classifier = FaceClassifier(bundle)

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.bundle.class_names

    def _single_face(self, image: Any, t: RecognitionThresholds, now: float):
        try:
            image = validate_image_array(image)
            gray = to_grayscale(image)
        except InvalidImageError as exc:
            return None, None, FrameAnalysis(FrameStatus.INVALID_IMAGE, (), None, str(exc), now)
        faces = self.detector.detect(gray, t.min_face_size_px)
        if not faces:
            return image, None, FrameAnalysis(FrameStatus.NO_FACE, (), None,
                                              "No face detected — face the camera with even lighting.", now)
        if len(faces) > 1:
            return image, None, FrameAnalysis(FrameStatus.MULTIPLE_FACES, tuple(faces), None,
                                              f"{len(faces)} faces detected — one student at a time, please.", now)
        return image, faces[0], None

    @staticmethod
    def _decide(face: FaceBox, prediction: Prediction, t: RecognitionThresholds, now: float) -> FrameAnalysis:
        reason = prediction.rejection_reason(t)
        if reason is None:
            return FrameAnalysis(FrameStatus.CANDIDATE, (face,), prediction, "Candidate in this frame.", now)
        return FrameAnalysis(FrameStatus.UNCERTAIN, (face,), prediction, f"Unknown: {reason}.", now)

    def analyze(self, image: Any, t: RecognitionThresholds) -> FrameAnalysis:
        """Analyse one BGR video frame. Never raises; problems are returned as a status."""
        now = time.monotonic()
        try:
            image, face, early = self._single_face(image, t, now)
            if early is not None:
                return early
            crop = crop_face(image, face, t.face_crop_margin)
            prediction = self.classifier.predict([bgr_to_pil(crop)])[0]
            return self._decide(face, prediction, t, now)
        except InvalidImageError as exc:
            return FrameAnalysis(FrameStatus.INVALID_IMAGE, (), None, str(exc), now)
        except Exception as exc:  # noqa: BLE001 - keep the video running and report the error
            return FrameAnalysis(FrameStatus.ERROR, (), None, f"Recognition error: {exc}", now)

    def analyze_still(self, image: Any, t: RecognitionThresholds, variants: int = 5) -> StillAnalysis:
        """Single photo: classify several slightly shifted crops; all must agree (replaces frame stability)."""
        now = time.monotonic()
        try:
            image, face, early = self._single_face(image, t, now)
            if early is not None:
                return StillAnalysis(early, (), 0.0, False)
            base = expand_box(face, t.face_crop_margin, image.shape)
            offsets = [(0.0, 0.0, 1.0), (-0.05, 0.0, 1.0), (0.05, 0.0, 1.0), (0.0, -0.05, 1.0), (0.0, 0.05, 1.0),
                       (0.0, 0.0, 0.92), (0.0, 0.0, 1.08), (-0.04, -0.04, 1.04), (0.04, 0.04, 0.96)]
            boxes = [shift_box(base, dx, dy, s, image.shape) for dx, dy, s in offsets[: max(1, variants)]]
            predictions = self.classifier.predict([bgr_to_pil(crop_box(image, b)) for b in boxes])
            main = predictions[0]
            analysis = self._decide(face, main, t, now)
            agreeing = sum(1 for p in predictions if p.label == main.label and p.passes(t))
            consistent = analysis.status is FrameStatus.CANDIDATE and agreeing == len(predictions)
            if analysis.status is FrameStatus.CANDIDATE and not consistent:
                analysis = FrameAnalysis(FrameStatus.UNCERTAIN, analysis.faces, main,
                                         f"Unknown: only {agreeing}/{len(predictions)} crop variants agree.", now)
            return StillAnalysis(analysis, tuple(predictions), agreeing / len(predictions), consistent)
        except InvalidImageError as exc:
            return StillAnalysis(FrameAnalysis(FrameStatus.INVALID_IMAGE, (), None, str(exc), now), (), 0.0, False)
        except Exception as exc:  # noqa: BLE001
            return StillAnalysis(FrameAnalysis(FrameStatus.ERROR, (), None, f"Recognition error: {exc}", now),
                                 (), 0.0, False)


# ======================================================================================
# 3. Stability across frames
# ======================================================================================


@dataclass(frozen=True)
class StabilitySnapshot:
    label: str | None
    consecutive: int
    required: int
    mean_confidence: float | None
    is_stable: bool
    event_id: int  # increases each time a NEW stable identity is reached

    @property
    def progress(self) -> float:
        return min(self.consecutive / self.required, 1.0) if self.required else 0.0


class RecognitionStabilizer:
    """Same identity in ``required_frames`` consecutive analysed frames, in roughly the same place.

    The streak restarts on: no face, several faces, an uncertain frame, a different identity, a
    pause longer than ``max_gap_s``, or a face box that jumped (IoU < ``min_iou``, the rule used by
    the notebook's live_attendance.py). Not thread-safe by itself; LiveRecognitionSession guards it.
    """

    def __init__(self, required_frames: int = 8, max_gap_s: float = 2.0, min_iou: float = 0.25) -> None:
        self.required_frames = max(1, int(required_frames))
        self.max_gap_s = float(max_gap_s)
        self.min_iou = float(min_iou)
        self._label: str | None = None
        self._box: FaceBox | None = None
        self._count = 0
        self._last_ts: float | None = None
        self._confidences: deque[float] = deque(maxlen=self.required_frames)
        self._was_stable = False
        self._event_id = 0

    def configure(self, required_frames: int, max_gap_s: float) -> None:
        required_frames = max(1, int(required_frames))
        if required_frames != self.required_frames:
            self.required_frames = required_frames
            self._confidences = deque(self._confidences, maxlen=required_frames)
        self.max_gap_s = float(max_gap_s)

    def reset(self) -> None:
        self._label, self._box, self._count, self._last_ts = None, None, 0, None
        self._confidences.clear()
        self._was_stable = False

    def update(self, analysis: FrameAnalysis) -> StabilitySnapshot:
        label = analysis.candidate_label
        if label is None:
            self.reset()
        else:
            assert analysis.prediction is not None
            box = analysis.face
            continuing = (
                label == self._label
                and self._last_ts is not None
                and analysis.timestamp - self._last_ts <= self.max_gap_s
                and (self._box is None or box is None or box_iou(self._box, box) >= self.min_iou)
            )
            if not continuing:
                self.reset()
                self._label = label
            self._count += 1
            self._box = box
            self._confidences.append(analysis.prediction.confidence)
            self._last_ts = analysis.timestamp
        return self.snapshot()

    def snapshot(self) -> StabilitySnapshot:
        stable = self._label is not None and self._count >= self.required_frames
        if stable and not self._was_stable:
            self._event_id += 1
        self._was_stable = stable
        values = list(self._confidences)
        return StabilitySnapshot(self._label, self._count, self.required_frames,
                                 float(np.mean(values)) if values else None, stable, self._event_id)
