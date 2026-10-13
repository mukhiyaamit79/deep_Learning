"""Typed access to the settings stored in the SQLite ``app_settings`` table."""

from __future__ import annotations

from typing import Any

from database.db import Database
from database.repositories import SettingsRepository
from services.face_recognition import RecognitionThresholds
from services.live_video import LiveConfig
from utils.config import SETTING_SPECS, default_settings


def load_settings(db: Database) -> dict[str, Any]:
    stored = SettingsRepository(db).get_all()
    values = default_settings()
    for key, spec in SETTING_SPECS.items():
        if key in stored:
            values[key] = spec.parse(stored[key])
    return values


def save_settings(db: Database, updates: dict[str, Any]) -> None:
    serialized: dict[str, str] = {}
    errors: list[str] = []
    for key, value in updates.items():
        spec = SETTING_SPECS.get(key)
        if spec is None:
            errors.append(f"Unknown setting '{key}'.")
            continue
        try:
            serialized[key] = spec.serialize(value)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError(" ".join(errors))
    SettingsRepository(db).set_many(serialized)


def reset_settings(db: Database) -> None:
    SettingsRepository(db).set_many({k: SETTING_SPECS[k].serialize(v) for k, v in default_settings().items()})


def recognition_thresholds(settings: dict[str, Any]) -> RecognitionThresholds:
    return RecognitionThresholds(
        confidence_threshold=float(settings["recognition.confidence_threshold"]),
        min_margin=float(settings["recognition.min_margin"]),
        min_face_size_px=int(settings["recognition.min_face_size_px"]),
        face_crop_margin=float(settings["recognition.face_crop_margin"]),
    )


def live_config(settings: dict[str, Any], overlay_names: dict[str, str]) -> LiveConfig:
    return LiveConfig(
        thresholds=recognition_thresholds(settings),
        stability_frames=int(settings["recognition.stability_frames"]),
        max_gap_s=int(settings["recognition.max_frame_gap_ms"]) / 1000.0,
        inference_interval_s=int(settings["recognition.inference_interval_ms"]) / 1000.0,
        show_boxes=bool(settings["display.show_face_boxes"]),
        mirror=bool(settings["camera.mirror_preview"]),
        overlay_names=tuple(sorted(overlay_names.items())),
    )
