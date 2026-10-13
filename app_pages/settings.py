"""Application settings (stored in SQLite), teacher accounts, file locations and system information."""

from __future__ import annotations

import importlib.metadata
import platform
from typing import Any

import pandas as pd
import streamlit as st

from database.db import DatabaseError
from services.attendance_service import AttendanceService
from services.auth_service import AuthService
from services.settings_service import load_settings, reset_settings, save_settings
from ui.auth_ui import require_teacher
from ui.components import page_header, session_label, show_validation_error
from ui.state import MODEL_LOAD_INFO, artifact_status, clear_model_cache, get_paths, require_database
from utils.config import (
    APP_CONFIG_FILE,
    DEFAULT_DATABASE_PATH,
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_DIR,
    SETTING_SPECS,
    SettingSpec,
    env_overrides_active,
    read_app_config,
    save_path_overrides,
)
from utils.validators import ValidationError

FLASH_KEY = "settings_flash"
GROUPS = {
    "recognition": ("Recognition", ":material/face_retouching_natural:"),
    "camera": ("Camera", ":material/videocam:"),
    "session": ("Attendance sessions", ":material/event:"),
    "display": ("Display", ":material/palette:"),
    "privacy": ("Privacy", ":material/shield:"),
}

user = require_teacher()
page_header("Settings", "Recognition thresholds, camera, sessions, display, privacy, accounts and file locations.")

db = require_database()
settings = load_settings(db)
if FLASH_KEY in st.session_state:
    st.success(st.session_state.pop(FLASH_KEY), icon=":material/check_circle:")

tab_app, tab_accounts, tab_paths, tab_system = st.tabs(
    [":material/tune: Application", ":material/manage_accounts: Teacher accounts", ":material/folder: Files & paths",
     ":material/memory: System"])


def widget(spec: SettingSpec, value: Any, open_sessions: list[dict[str, Any]]) -> Any:
    key = f"set_{spec.key}"
    if spec.key == "session.active_session_id":
        by_id = {0: None, **{int(s["id"]): s for s in open_sessions}}
        ids = list(by_id)
        return st.selectbox(spec.label, ids, index=ids.index(int(value)) if int(value) in by_id else 0,
                            format_func=lambda k: "Latest running session" if k == 0 else session_label(by_id[k]),
                            help=spec.help, key=key)
    if spec.kind is bool:
        return st.toggle(spec.label, value=bool(value), help=spec.help or None, key=key)
    if spec.choices:
        options = list(spec.choices)
        return st.selectbox(spec.label, options, index=options.index(value) if value in options else 0,
                            help=spec.help or None, key=key)
    if spec.kind is str:
        return st.text_input(spec.label, value=str(value), help=spec.help or None, key=key)
    if spec.kind is float:
        return st.slider(spec.label, min_value=float(spec.minimum), max_value=float(spec.maximum), value=float(value),
                         step=0.01, help=spec.help or None, key=key)
    if spec.minimum is not None and spec.maximum is not None and spec.maximum - spec.minimum <= 100:
        return st.slider(spec.label, min_value=int(spec.minimum), max_value=int(spec.maximum), value=int(value), step=1,
                         help=spec.help or None, key=key)
    return st.number_input(spec.label, min_value=int(spec.minimum) if spec.minimum is not None else None,
                           max_value=int(spec.maximum) if spec.maximum is not None else None, value=int(value),
                           step=10 if (spec.maximum or 0) > 1000 else 1, help=spec.help or None, key=key)


with tab_app:
    st.caption("Saved in the SQLite app_settings table and applied immediately. Defaults follow the notebook "
               "(threshold 0.90, 8 consecutive frames).")
    open_sessions = AttendanceService(db).open_sessions()
    with st.form("settings_form"):
        values: dict[str, Any] = {}
        for prefix, (title, icon) in GROUPS.items():
            st.markdown(f"##### {icon} {title}")
            specs = [s for k, s in SETTING_SPECS.items() if k.startswith(prefix + ".")]
            cols = st.columns(2)
            for i, spec in enumerate(specs):
                with cols[i % 2]:
                    values[spec.key] = widget(spec, settings[spec.key], open_sessions)
        saved = st.form_submit_button("Save settings", type="primary", icon=":material/save:")
    if saved:
        try:
            save_settings(db, values)
        except (ValueError, DatabaseError) as exc:
            st.error(str(exc))
        else:
            st.session_state[FLASH_KEY] = "Settings saved."
            st.rerun()
    if st.button("Restore defaults", icon=":material/restart_alt:", key="settings_reset"):
        reset_settings(db)
        st.session_state[FLASH_KEY] = "Default settings restored."
        st.rerun()

with tab_accounts:
    auth = AuthService(db)
    accounts = pd.DataFrame(auth.list_accounts())
    teachers = accounts[accounts["role"] == "teacher"] if not accounts.empty else accounts
    st.dataframe(teachers[["username", "display_name", "last_login_at"]] if not teachers.empty else teachers,
                 hide_index=True)
    st.caption("Student logins are created automatically for every registered student (Students page).")
    with st.form("add_teacher", clear_on_submit=True):
        st.markdown("##### Add a teacher account")
        c1, c2 = st.columns(2)
        username = c1.text_input("Username")
        display = c2.text_input("Display name")
        c3, c4 = st.columns(2)
        p1 = c3.text_input("Password", type="password")
        p2 = c4.text_input("Repeat password", type="password")
        add = st.form_submit_button("Add teacher", icon=":material/person_add:")
    if add:
        try:
            auth.create_teacher(username, display, p1, p2)
        except (ValidationError, DatabaseError) as exc:
            show_validation_error(exc)
        else:
            st.session_state[FLASH_KEY] = f"Teacher account '{username.strip()}' added."
            st.rerun()

with tab_paths:
    paths = get_paths()
    st.table(pd.DataFrame({"Item": list(paths.display_dict()), "Location": list(paths.display_dict().values())}))
    if env_overrides_active():
        st.info("Environment variables override app_config.json: " + ", ".join(env_overrides_active()))
    cfg = read_app_config()
    with st.form("paths_form"):
        st.caption(f"Saved to {APP_CONFIG_FILE.name}. Relative paths are relative to the project folder. Changing the "
                   "database path does not move or delete the old database — restart the app afterwards.")
        models_dir = st.text_input("Models folder", value=str(cfg.get("models_dir", DEFAULT_MODELS_DIR)))
        results_dir = st.text_input("Results folder", value=str(cfg.get("results_dir", DEFAULT_RESULTS_DIR)))
        database_path = st.text_input("SQLite database", value=str(cfg.get("database_path", DEFAULT_DATABASE_PATH)))
        save_paths = st.form_submit_button("Save locations", icon=":material/save:")
    if save_paths:
        try:
            save_path_overrides(models_dir=models_dir, results_dir=results_dir, database_path=database_path)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
        else:
            clear_model_cache()
            st.session_state[FLASH_KEY] = "Locations saved. Restart the app if you changed the database path."
            st.rerun()
    status = artifact_status()
    st.markdown("##### Model")
    if MODEL_LOAD_INFO:
        st.caption(f"Loaded {MODEL_LOAD_INFO['loaded_at']} on {MODEL_LOAD_INFO['device']} in "
                   f"{MODEL_LOAD_INFO['load_seconds']} s · {MODEL_LOAD_INFO['classes']} classes.")
    else:
        st.caption("Not loaded in this server process yet.")
    if not status.ready:
        st.caption("Model files: " + "; ".join(status.errors))
    if st.button("Reload model from disk", icon=":material/refresh:", key="reload_model"):
        clear_model_cache()
        st.session_state[FLASH_KEY] = "Model cache cleared; it reloads on the Live Attendance page."
        st.rerun()

with tab_system:
    def version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    packages = ["streamlit", "streamlit-webrtc", "torch", "opencv-python", "opencv-python-headless", "numpy", "pandas",
                "Pillow", "plotly", "av", "aiortc"]
    cuda = "unknown"
    try:
        import torch

        cuda = f"available ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "not available — CPU is used"
    except Exception:  # noqa: BLE001
        cuda = "PyTorch not importable"
    st.table(pd.DataFrame({"Component": ["Python", "Platform", "CUDA for the web app", *packages],
                           "Version": [platform.python_version(), platform.platform(terse=True), cuda,
                                       *[version(p) for p in packages]]}))
