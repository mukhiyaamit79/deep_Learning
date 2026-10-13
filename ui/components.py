"""Reusable Streamlit building blocks."""

from __future__ import annotations

import html
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from database.repositories import DuplicateSessionError
from services.attendance_service import AttendanceService
from services.face_recognition import ArtifactStatus
from ui.theme import COURSE, STATUS_LABELS, status_badge
from utils.validators import ValidationError

NOTEBOOK_EXPORT_STEPS = """
1. Open **`amit_eda.ipynb`**, select the **Anaconda base** kernel, restart it and press **Run All**.
2. Cell **NN-36** saves `models/attendance_cnn.pt`, `class_names.json`, `preprocessing_config.json`
   and `model_metadata.json`; cell **NN-39** writes the `results/` files.
3. Refresh this page — the app picks the new files up automatically.
"""


def page_header(title: str, subtitle: str | None = None, *, kicker: str | None = None) -> None:
    kicker_text = html.escape(kicker or f"{COURSE} · Smart Attendance System")
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="sas-header"><div class="sas-kicker">{kicker_text}</div>'
                f"<h1>{html.escape(title)}</h1>{sub}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: Any, *, help: str | None = None, delta: str | None = None,
                delta_color: str = "normal") -> None:
    with st.container(border=True):
        st.metric(label, value, delta=delta, help=help, delta_color=delta_color)


def empty_state(title: str, body: str, icon: str = ":material/inbox:") -> None:
    with st.container(border=True):
        st.markdown(f"#### {icon} {title}")
        st.caption(body)


def format_confidence(value: float | None, as_percent: bool = True) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1%}" if as_percent else f"{float(value):.3f}"


def format_rate(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.0%}"


def status_text(status: str | None) -> str:
    return STATUS_LABELS.get(status or "", status or "—")


def clock(iso_value: Any) -> str:
    """'2026-10-11T09:12:30' -> '09:12'."""
    text = str(iso_value or "")
    return text[11:16] if len(text) >= 16 else text


def session_label(session: dict[str, Any]) -> str:
    section = f" / {session['section']}" if session.get("section") else ""
    start = f" {session['start_time']}" if session.get("start_time") else ""
    state = "running" if session.get("status") == "open" else "ended"
    return f"{session['session_date']}{start} · {session['session_name']} · {session['class_name']}{section} ({state})"


def show_validation_error(exc: Exception) -> None:
    if isinstance(exc, ValidationError):
        st.error("Please fix the following:\n\n" + "\n".join(f"- {e}" for e in exc.errors))
    else:
        st.error(str(exc))


def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV", key: str | None = None) -> None:
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv",
                       icon=":material/download:", disabled=df.empty, key=key)


def artifact_problem_box(status: ArtifactStatus) -> None:
    st.warning("**The trained model is not available, so face recognition is disabled.**\n\n"
               + "\n".join(f"- {e}" for e in status.errors), icon=":material/model_training:")
    with st.expander("How to produce the model files", expanded=True):
        st.markdown(NOTEBOOK_EXPORT_STEPS)


def avatar(photo: Path | None, name: str) -> None:
    """Student photo (from dataset/) or an initials placeholder."""
    if photo is not None and photo.is_file():
        st.image(str(photo), width="stretch")
    else:
        initials = "".join(part[0] for part in name.split()[:2]).upper() or "?"
        st.markdown(f'<div class="sas-avatar">{html.escape(initials)}</div>', unsafe_allow_html=True)


def student_card(student: dict[str, Any], photo: Path | None, status: str | None, detail: str | None = None) -> None:
    with st.container(border=True):
        avatar(photo, str(student.get("full_name") or "?"))
        st.markdown(f'<div class="sas-name">{html.escape(str(student.get("full_name")))}</div>'
                    f'<div class="sas-muted">{html.escape(str(student.get("student_id")))}</div>',
                    unsafe_allow_html=True)
        st.markdown(status_badge(status, detail), unsafe_allow_html=True)


def session_create_form(service: AttendanceService, operator: str, *, key: str, class_options: list[str],
                        default_class: str | None = None) -> int | None:
    """Full session form. Returns the new session id when created."""
    with st.form(key, clear_on_submit=False):
        c1, c2 = st.columns(2)
        name = c1.text_input("Session name", value=f"{default_class or 'Class'} lecture", max_chars=80)
        day = c2.date_input("Date", value=date.today())
        c3, c4 = st.columns(2)
        options = class_options or ([default_class] if default_class else [])
        if options:
            index = options.index(default_class) if default_class in options else 0
            class_name = c3.selectbox("Class", options=options, index=index)
        else:
            class_name = c3.text_input("Class", placeholder="e.g. TECH 405")
        section = c4.text_input("Section (optional)", max_chars=20, help="Empty = every section of the class.")
        c5, c6 = st.columns(2)
        now = datetime.now()
        start = c5.time_input("Start time", value=time(now.hour, (now.minute // 5) * 5), step=300)
        end = c6.time_input("End time", value=time(min(now.hour + 1, 23), (now.minute // 5) * 5), step=300)
        submitted = st.form_submit_button("Create session", type="primary", icon=":material/event_available:")
    if not submitted:
        return None
    try:
        session_id = service.create_session(session_name=name, session_date=day, class_name=class_name or "",
                                            section=section, start_time=start, end_time=end, operator=operator)
    except (ValidationError, DuplicateSessionError) as exc:
        show_validation_error(exc)
        return None
    st.success("Session created.", icon=":material/check_circle:")
    return session_id
