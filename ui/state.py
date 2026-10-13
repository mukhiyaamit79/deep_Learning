"""Process-wide cached resources.

* The database is created / migrated and the demo data seeded once per server process.
* The PyTorch model is loaded once and shared by every page, browser session and video frame
  (``st.cache_resource``). The cache key includes the model files' modification times, so
  re-running notebook NN-36 automatically reloads the new model.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import streamlit as st

from database import Database, DatabaseError, InitResult, initialize_database
from services.face_recognition import ArtifactStatus, FaceDetector, FaceRecognizer, inspect_artifacts, load_model_bundle
from services.seed import SeedResult, seed_demo_data
from utils.config import AppPaths, load_paths

MODEL_LOAD_INFO: dict[str, Any] = {}


def get_paths() -> AppPaths:
    return load_paths()


@st.cache_resource(show_spinner="Preparing the local SQLite database…")
def _initialise(db_path: str, backups_dir: str) -> tuple[InitResult, SeedResult]:
    init = initialize_database(Path(db_path), Path(backups_dir))
    seed = seed_demo_data(Database(Path(db_path)), get_paths())
    return init, seed


def get_database() -> tuple[Database, InitResult, SeedResult]:
    paths = get_paths()
    init, seed = _initialise(str(paths.database_path), str(paths.backups_dir))
    return Database(paths.database_path), init, seed


def require_database() -> Database:
    try:
        db, _, _ = get_database()
        return db
    except (DatabaseError, sqlite3.Error, OSError) as exc:
        st.error(f"**The attendance database is unavailable.**\n\n{exc}", icon=":material/database:")
        st.caption("The database is never deleted or recreated automatically. Check Settings → Files & paths.")
        st.stop()
        raise


def artifact_status() -> ArtifactStatus:
    return inspect_artifacts(get_paths())


@st.cache_resource(show_spinner="Loading the trained CNN (first time only)…", max_entries=1)
def _load_recognizer(models_dir: str, results_dir: str, database_path: str, fingerprint: tuple) -> FaceRecognizer:
    paths = AppPaths(models_dir=Path(models_dir), results_dir=Path(results_dir), database_path=Path(database_path))
    bundle = load_model_bundle(paths)
    recognizer = FaceRecognizer(bundle, FaceDetector())
    MODEL_LOAD_INFO.clear()
    MODEL_LOAD_INFO.update({
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "load_seconds": round(bundle.load_seconds, 2),
        "device": bundle.device, "torch_version": bundle.torch_version, "parameters": bundle.parameter_count,
        "classes": len(bundle.class_names), "fingerprint": fingerprint,
    })
    return recognizer


def get_recognizer(status: ArtifactStatus | None = None) -> FaceRecognizer:
    """Load (or reuse) the recognizer. Raises ModelArtifactError with an actionable message."""
    paths = get_paths()
    status = status or inspect_artifacts(paths)
    return _load_recognizer(str(paths.models_dir), str(paths.results_dir), str(paths.database_path), status.fingerprint)


def model_loaded(status: ArtifactStatus) -> bool:
    return bool(MODEL_LOAD_INFO) and MODEL_LOAD_INFO.get("fingerprint") == status.fingerprint


def clear_model_cache() -> None:
    _load_recognizer.clear()
    MODEL_LOAD_INFO.clear()
