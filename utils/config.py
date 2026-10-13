"""Central configuration for the Smart Attendance web app.

* Project paths (derived from this file's location with pathlib — no absolute paths).
* The notebook -> app file contract: the app only READS what ``amit_eda.ipynb`` writes in
  NN-36 (models/) and NN-39 (results/). It never trains a model.
* Demo login accounts (academic demonstration only).
* Definitions, defaults and valid ranges of the settings stored in SQLite.

Path resolution order: environment variables > app_config.json > defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------------------------

#: Project folder: the one that contains app.py, amit_eda.ipynb and attendance_utils.py.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
APP_CONFIG_FILE: Path = PROJECT_ROOT / "app_config.json"

DEFAULT_DATASET_DIR = "dataset"
DEFAULT_MODELS_DIR = "models"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_DATABASE_PATH = "database_storage/attendance.db"

ENV_DB_PATH = "SMART_ATTENDANCE_DB_PATH"
ENV_MODELS_DIR = "SMART_ATTENDANCE_MODELS_DIR"
ENV_RESULTS_DIR = "SMART_ATTENDANCE_RESULTS_DIR"

# --------------------------------------------------------------------------------------
# Files written by amit_eda.ipynb (NN-36: models/, NN-39: results/)
# --------------------------------------------------------------------------------------

MODEL_FILENAME = "attendance_cnn.pt"
CLASS_NAMES_FILENAME = "class_names.json"
PREPROCESSING_FILENAME = "preprocessing_config.json"
METADATA_FILENAME = "model_metadata.json"

COMPARISON_CSV = "model_comparison.csv"
PER_CLASS_CSV = "per_class_metrics.csv"
EXPERIMENTS_CSV = "experiment_configurations.csv"
VALIDATION_CSV = "validation_comparison.csv"
SUMMARY_JSON = "summary.json"
FIGURES_DIRNAME = "figures"

MODEL_NAMES: tuple[str, ...] = ("CNN", "RNN", "LSTM")
COMPARISON_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Model", "Accuracy", "Precision", "Recall", "F1-score",
    "Parameters", "Training Time (s)", "Inference Time (ms/image)",
)

#: Figures exported by the notebook into results/figures/ (only these are shown on the
#: comparison page, so unrelated images elsewhere in results/ are never mixed in).
FIGURE_TITLES: dict[str, str] = {
    "history_accuracy.png": "Training accuracy per epoch",
    "history_val_accuracy.png": "Validation accuracy per epoch",
    "history_loss.png": "Training loss per epoch",
    "history_val_loss.png": "Validation loss per epoch",
    "confusion_counts.png": "Confusion matrices (counts)",
    "confusion_normalized.png": "Confusion matrices (normalised)",
    "test_accuracy.png": "Test accuracy",
    "test_precision.png": "Test precision",
    "test_recall.png": "Test recall",
    "test_f1-score.png": "Test F1-score",
    "training_time.png": "Training time",
    "parameters.png": "Parameter count",
    "inference_latency.png": "Inference latency",
    "batch_throughput.png": "Batch throughput",
}

# --------------------------------------------------------------------------------------
# Demo accounts — ACADEMIC DEMONSTRATION ONLY
# --------------------------------------------------------------------------------------
# Created once, on the first start, as salted password hashes in SQLite. Change them on the
# "My account" page. Never reuse these passwords anywhere real.

DEMO_TEACHER_USERNAME = "teacher"
DEMO_TEACHER_PASSWORD = "teacher123"
DEMO_TEACHER_NAME = "Course Teacher"
#: Every student logs in with their Student ID (e.g. "Amit") and this password.
DEMO_STUDENT_PASSWORD = "student123"
#: Class assigned to the students created from models/class_names.json on first start.
DEFAULT_CLASS_NAME = "TECH 405"


def _resolve(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class AppPaths:
    """All file-system locations used by the application."""

    models_dir: Path
    results_dir: Path
    database_path: Path
    dataset_dir: Path = field(default=PROJECT_ROOT / DEFAULT_DATASET_DIR)
    project_root: Path = field(default=PROJECT_ROOT)

    @property
    def model_path(self) -> Path:
        return self.models_dir / MODEL_FILENAME

    @property
    def class_names_path(self) -> Path:
        return self.models_dir / CLASS_NAMES_FILENAME

    @property
    def preprocessing_path(self) -> Path:
        return self.models_dir / PREPROCESSING_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / METADATA_FILENAME

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / FIGURES_DIRNAME

    @property
    def comparison_csv(self) -> Path:
        return self.results_dir / COMPARISON_CSV

    @property
    def per_class_csv(self) -> Path:
        return self.results_dir / PER_CLASS_CSV

    @property
    def experiments_csv(self) -> Path:
        return self.results_dir / EXPERIMENTS_CSV

    @property
    def validation_csv(self) -> Path:
        return self.results_dir / VALIDATION_CSV

    @property
    def summary_json(self) -> Path:
        return self.results_dir / SUMMARY_JSON

    @property
    def backups_dir(self) -> Path:
        return self.database_path.parent / "backups"

    def display_dict(self) -> dict[str, str]:
        def rel(p: Path) -> str:
            try:
                return str(p.relative_to(self.project_root))
            except ValueError:
                return str(p)

        return {
            "Project folder": str(self.project_root),
            "Dataset (student photos)": rel(self.dataset_dir),
            "Model files": rel(self.models_dir),
            "Notebook results": rel(self.results_dir),
            "SQLite database": rel(self.database_path),
        }


def read_app_config() -> dict[str, Any]:
    """Read app_config.json if present; an invalid file is ignored (startup never crashes)."""
    if not APP_CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(APP_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_paths() -> AppPaths:
    cfg = read_app_config()
    models = os.environ.get(ENV_MODELS_DIR) or cfg.get("models_dir") or DEFAULT_MODELS_DIR
    results = os.environ.get(ENV_RESULTS_DIR) or cfg.get("results_dir") or DEFAULT_RESULTS_DIR
    database = os.environ.get(ENV_DB_PATH) or cfg.get("database_path") or DEFAULT_DATABASE_PATH
    return AppPaths(
        models_dir=_resolve(str(models)),
        results_dir=_resolve(str(results)),
        database_path=_resolve(str(database)),
    )


def save_path_overrides(*, models_dir: str, results_dir: str, database_path: str) -> None:
    """Persist path overrides to app_config.json (atomic write)."""
    for label, value in (("Models folder", models_dir), ("Results folder", results_dir),
                         ("Database path", database_path)):
        if not str(value).strip():
            raise ValueError(f"{label} cannot be empty.")
    if not str(database_path).lower().endswith((".db", ".sqlite", ".sqlite3")):
        raise ValueError("Database path must end with .db, .sqlite or .sqlite3.")
    cfg = read_app_config()
    cfg.update({"models_dir": str(models_dir).strip(), "results_dir": str(results_dir).strip(),
                "database_path": str(database_path).strip()})
    tmp = APP_CONFIG_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    os.replace(tmp, APP_CONFIG_FILE)


def env_overrides_active() -> list[str]:
    return [name for name in (ENV_DB_PATH, ENV_MODELS_DIR, ENV_RESULTS_DIR) if os.environ.get(name)]


# --------------------------------------------------------------------------------------
# Settings stored in SQLite (app_settings table)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SettingSpec:
    key: str
    kind: type
    default: Any
    label: str
    help: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] | None = None

    def coerce(self, value: Any) -> Any:
        """Convert ``value`` to this setting's type and validate it. Raises ValueError."""
        if self.kind is bool:
            if isinstance(value, bool):
                result: Any = value
            elif str(value).strip().lower() in {"1", "true", "yes", "on"}:
                result = True
            elif str(value).strip().lower() in {"0", "false", "no", "off"}:
                result = False
            else:
                raise ValueError(f"{self.label}: expected true/false, got {value!r}.")
        elif self.kind is int:
            try:
                result = int(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.label}: expected a whole number, got {value!r}.") from exc
        elif self.kind is float:
            try:
                result = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.label}: expected a number, got {value!r}.") from exc
        else:
            result = str(value).strip()
            if not result:
                raise ValueError(f"{self.label}: cannot be empty.")
        if self.minimum is not None and result < self.minimum:
            raise ValueError(f"{self.label}: must be ≥ {self.minimum}.")
        if self.maximum is not None and result > self.maximum:
            raise ValueError(f"{self.label}: must be ≤ {self.maximum}.")
        if self.choices is not None and result not in self.choices:
            raise ValueError(f"{self.label}: must be one of {', '.join(map(str, self.choices))}.")
        return result

    def serialize(self, value: Any) -> str:
        coerced = self.coerce(value)
        if self.kind is bool:
            return "true" if coerced else "false"
        return str(coerced)

    def parse(self, raw: str | None) -> Any:
        if raw is None:
            return self.default
        try:
            return self.coerce(raw)
        except ValueError:
            return self.default


_SPECS: tuple[SettingSpec, ...] = (
    # ---- Recognition (defaults = the notebook's preprocessing_config.json policy) ----
    SettingSpec("recognition.auto_mark", bool, True, "Auto-mark Present when a face is recognised",
                "When ON, a student is marked Present automatically once the same identity has been "
                "recognised in the required number of consecutive frames. The teacher can undo it. "
                "When OFF, the teacher clicks Confirm for every student."),
    SettingSpec("recognition.confidence_threshold", float, 0.90, "Confidence threshold",
                "Minimum softmax score of the top student (the notebook uses 0.90). A high score does "
                "not prove identity: the model always picks one of the trained students.", 0.30, 0.99),
    SettingSpec("recognition.stability_frames", int, 8, "Consecutive frames required",
                "The same student must be recognised in this many analysed frames in a row "
                "(the notebook's live script uses 8).", 2, 60),
    SettingSpec("recognition.min_margin", float, 0.0, "Minimum lead over the 2nd student",
                "Optional extra check: top score minus second score. 0 = off (notebook policy).", 0.0, 0.95),
    SettingSpec("recognition.face_crop_margin", float, 0.0, "Extra margin around the detected face",
                "Training photos are tight 200×200 face crops, so 0.0 matches them best.", 0.0, 0.5),
    SettingSpec("recognition.min_face_size_px", int, 60, "Minimum face size (pixels)",
                "Faces smaller than this are ignored (the notebook's live script uses 60).", 30, 400),
    SettingSpec("recognition.inference_interval_ms", int, 120, "Time between analysed frames (ms)",
                "Throttles detection + CNN so the video stays smooth.", 0, 2000),
    SettingSpec("recognition.max_frame_gap_ms", int, 2000, "Maximum pause between agreeing frames (ms)",
                "A longer pause restarts the consecutive-frame count.", 200, 10000),
    SettingSpec("recognition.still_image_variants", int, 5, "Still-photo crop variants",
                "For a single photo, the face is classified under several slightly shifted crops; "
                "all must agree.", 1, 9),
    # ---- Camera ----
    SettingSpec("camera.resolution", str, "640x480", "Camera resolution (ideal)", "",
                choices=("320x240", "640x480", "800x600", "1280x720")),
    SettingSpec("camera.frame_rate", int, 15, "Camera frame rate (ideal)", "", 5, 30),
    SettingSpec("camera.mirror_preview", bool, True, "Mirror the live preview",
                "Display only. Recognition always uses the original frame."),
    SettingSpec("camera.use_stun_server", bool, False, "Use a public STUN server",
                "Not needed on localhost. Turn ON only if the camera stays on 'Connecting…'. "
                "Only connection metadata is exchanged; video stays on this computer."),
    # ---- Sessions ----
    SettingSpec("session.default_class_name", str, DEFAULT_CLASS_NAME, "Default class for new sessions"),
    SettingSpec("session.active_session_id", int, 0, "Default attendance session",
                "Pre-selected open session on the Live Attendance page (0 = latest).", 0, None),
    SettingSpec("session.mark_late", bool, False, "Mark 'Late' after the start time",
                "When ON, students recognised more than the minutes below after the session start "
                "are marked Late instead of Present."),
    SettingSpec("session.late_after_minutes", int, 10, "Late after (minutes past start)", "", 0, 240),
    # ---- Display ----
    SettingSpec("display.show_face_boxes", bool, True, "Draw face boxes on the video"),
    SettingSpec("display.show_top_predictions", bool, False, "Show the top-3 model scores",
                "Useful to explain the CNN during a demo."),
    SettingSpec("display.confidence_as_percent", bool, True, "Show confidence as a percentage"),
    SettingSpec("display.recent_records_limit", int, 8, "Recent records on the dashboard", "", 3, 50),
    # ---- Privacy ----
    SettingSpec("privacy.log_recognition_events", bool, True, "Log recognition events",
                "Stores marked / duplicate / undone / rejected events (label, score, outcome). "
                "Face images are never stored."),
)

SETTING_SPECS: dict[str, SettingSpec] = {spec.key: spec for spec in _SPECS}


def default_settings() -> dict[str, Any]:
    return {key: spec.default for key, spec in SETTING_SPECS.items()}


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width_s, height_s = str(value).lower().split("x", 1)
        return int(width_s), int(height_s)
    except ValueError:
        return 640, 480
