"""Attendance records, sessions, manual entries, corrections and audit trail."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from database.db import DatabaseError
from database.repositories import (
    ATTENDANCE_METHODS,
    ATTENDANCE_STATUSES,
    AttendanceRepository,
    RecognitionLogRepository,
    SessionRepository,
)
from services.attendance_service import AttendanceService
from services.reporting_service import attendance_rate
from services.settings_service import load_settings
from services.student_service import StudentService
from ui.auth_ui import require_teacher
from ui.components import (
    clock,
    download_csv,
    empty_state,
    format_rate,
    metric_card,
    page_header,
    session_create_form,
    session_label,
    show_validation_error,
    status_text,
)
from ui.state import get_paths, require_database
from utils.validators import ValidationError

FLASH_KEY = "rec_flash"
METHOD_LABELS = {"face_recognition": "Camera", "manual": "Manual", "session_finalize": "Session ended"}

user = require_teacher()
page_header("Attendance Records", "Every attendance mark stored in SQLite — filter, export, correct and audit.")

db = require_database()
paths = get_paths()
settings = load_settings(db)
service = AttendanceService(db, log_events=bool(settings["privacy.log_recognition_events"]))
records_repo = AttendanceRepository(db)
sessions_repo = SessionRepository(db)
students = StudentService(db, paths.dataset_dir)

if FLASH_KEY in st.session_state:
    kind, message = st.session_state.pop(FLASH_KEY)
    getattr(st, kind)(message)


def date_range(label: str, key: str, days: int = 30) -> tuple[date, date]:
    today = date.today()
    value = st.date_input(label, value=(today - timedelta(days=days), today), key=key, format="YYYY-MM-DD")
    if isinstance(value, (tuple, list)):
        return (value[0], value[-1]) if value else (today, today)
    return value, value


tab_records, tab_sessions, tab_manual, tab_audit = st.tabs(
    [":material/table_view: Records", ":material/event: Sessions", ":material/edit_note: Manual entry",
     ":material/history: Corrections & logs"])

# =====================================================================================
# Records
# =====================================================================================
with tab_records:
    f1, f2, f3 = st.columns([1.4, 1, 1])
    with f1:
        d_from, d_to = date_range("Session dates", "rec_dates")
    class_choice = f2.selectbox("Class", ["All classes", *students.class_names()], key="rec_class")
    search = f3.text_input("Search", placeholder="Student, session or teacher", key="rec_search")
    f4, f5, f6 = st.columns([1, 1, 1.4])
    status_filter = f4.multiselect("Status", list(ATTENDANCE_STATUSES), format_func=status_text, key="rec_status")
    method_filter = f5.multiselect("Source", list(ATTENDANCE_METHODS), format_func=lambda m: METHOD_LABELS.get(m, m),
                                   key="rec_method")
    everyone = students.all_students(active=None)
    names = {int(s["id"]): f"{s['student_id']} · {s['full_name']}" for s in everyone}
    student_choice = f6.selectbox("Student", [None, *names], format_func=lambda k: "All students" if k is None else names[k],
                                  key="rec_student")

    rows = records_repo.search(date_from=d_from.isoformat(), date_to=d_to.isoformat(),
                               class_name=None if class_choice == "All classes" else class_choice,
                               statuses=status_filter, methods=method_filter, student_pk=student_choice,
                               search=search or None)
    df = pd.DataFrame(rows)
    counts = {s: int((df["status"] == s).sum()) if not df.empty else 0 for s in ATTENDANCE_STATUSES}
    m = st.columns(5)
    for col, (label, value) in zip(m, [("Records", len(df)), ("Present", counts["present"]), ("Late", counts["late"]),
                                       ("Absent", counts["absent"]),
                                       ("Attendance rate", format_rate(attendance_rate(counts["present"] + counts["late"],
                                                                                       counts["absent"])))]):
        with col:
            metric_card(label, value)

    if df.empty:
        empty_state("No records match the filters", "Widen the date range or clear filters.", ":material/filter_alt_off:")
    else:
        view = df.copy()
        view["status"] = view["status"].map(status_text)
        view["method"] = view["method"].map(lambda v: METHOD_LABELS.get(v, v))
        view["time"] = view["recorded_at"].map(clock)
        view["confidence"] = view["confidence"].map(lambda v: None if pd.isna(v) else round(float(v) * 100, 1))
        cols = ["attendance_id", "session_date", "time", "session_name", "student_id", "full_name", "status", "method",
                "confidence", "confirmed_by", "updated_at", "notes"]
        st.dataframe(view[cols], hide_index=True, column_config={
            "attendance_id": st.column_config.NumberColumn("ID", format="%d"), "session_date": "Date", "time": "Time",
            "session_name": "Session", "student_id": "Student ID", "full_name": "Name", "status": "Status",
            "method": "Source", "confidence": st.column_config.NumberColumn("Score", format="%.1f%%"),
            "confirmed_by": "Marked by", "updated_at": "Last corrected", "notes": "Notes"})
        download_csv(df.drop(columns=["photo_path"], errors="ignore"), f"attendance_{d_from}_{d_to}.csv",
                     "Export filtered records (CSV)", key="rec_csv")

        st.markdown("##### Correct a record")
        options = {int(r["attendance_id"]): r for r in rows}
        with st.form("correct_record"):
            c1, c2 = st.columns([2, 1])
            target = c1.selectbox("Record", list(options), format_func=lambda k: (
                f"#{k} · {options[k]['session_date']} · {options[k]['session_name']} · {options[k]['full_name']} "
                f"({status_text(options[k]['status'])})"))
            new_status = c2.selectbox("New status", list(ATTENDANCE_STATUSES), format_func=status_text)
            reason = st.text_input("Reason (required)", max_chars=200, placeholder="e.g. wrong match; medical note")
            submitted = st.form_submit_button("Apply correction", type="primary", icon=":material/edit:")
        if submitted:
            try:
                changed = service.correct(target, new_status=new_status, reason=reason, operator=user.username)
            except (ValidationError, DatabaseError) as exc:
                show_validation_error(exc)
            else:
                if changed:
                    st.session_state[FLASH_KEY] = ("success", f"Record #{target} changed to {status_text(new_status)} "
                                                              "(logged in the audit trail).")
                    st.rerun()
                st.info("The record already has that status.")

# =====================================================================================
# Sessions
# =====================================================================================


@st.dialog("End attendance session")
def confirm_end(session: dict, operator: str) -> None:
    st.write(f"**{session_label(session)}**")
    st.write(f"{session['not_marked_count']} student(s) without a mark will be recorded as **absent**.")
    c1, c2 = st.columns(2)
    if c1.button("End session", type="primary", key="dlg_rec_end_yes"):
        try:
            inserted = service.close_session(int(session["id"]), operator=operator)
        except DatabaseError as exc:
            st.error(str(exc))
            return
        st.session_state[FLASH_KEY] = ("success", f"Session ended: {inserted} absence record(s) created.")
        st.rerun()
    if c2.button("Cancel", key="dlg_rec_end_no"):
        st.rerun()


with tab_sessions:
    with st.expander("Create a session", icon=":material/add:"):
        if session_create_form(service, user.username, key="rec_new_session", class_options=students.class_names(),
                               default_class=str(settings["session.default_class_name"])):
            st.rerun()
    s1, s2 = st.columns([1.4, 1])
    with s1:
        sd_from, sd_to = date_range("Session dates", "sess_dates")
    state = s2.selectbox("State", ["All", "Running", "Ended"], key="sess_state")
    sessions = sessions_repo.list(date_from=sd_from.isoformat(), date_to=sd_to.isoformat(),
                                  status={"All": None, "Running": "open", "Ended": "closed"}[state])
    if not sessions:
        empty_state("No sessions in this range", "Start one on Live Attendance or create one above.", ":material/event_busy:")
    else:
        sdf = pd.DataFrame(sessions)
        sdf["state"] = sdf["status"].map({"open": "Running", "closed": "Ended"})
        st.dataframe(sdf[["id", "session_date", "start_time", "session_name", "class_name", "state", "roster_count",
                          "attended_count", "absent_count", "not_marked_count"]], hide_index=True, column_config={
            "id": st.column_config.NumberColumn("ID", format="%d"), "session_date": "Date", "start_time": "Start",
            "session_name": "Session", "class_name": "Class", "state": "State", "roster_count": "Roster",
            "attended_count": "Present + late", "absent_count": "Absent",
            "not_marked_count": st.column_config.NumberColumn("Not yet marked",
                                                              help="Not absent until the session is ended.")})
        by_id = {int(s["id"]): s for s in sessions}
        chosen = st.selectbox("Session details", list(by_id), format_func=lambda k: session_label(by_id[k]),
                              key="sess_detail")
        session = by_id[chosen]
        roster = pd.DataFrame(sessions_repo.roster(chosen))
        if roster.empty:
            st.caption("No active students match this session's class.")
        else:
            roster["status"] = roster["status"].map(lambda v: status_text(v) if isinstance(v, str) else "Not yet marked")
            roster["time"] = roster["recorded_at"].map(clock)
            st.dataframe(roster[["student_id", "full_name", "status", "method", "time", "confirmed_by"]], hide_index=True,
                         column_config={"student_id": "Student ID", "full_name": "Name", "status": "Status",
                                        "method": "Source", "time": "Time", "confirmed_by": "Marked by"})
        if session["status"] == "open":
            if st.button("End session…", icon=":material/stop_circle:", key=f"end_{chosen}"):
                confirm_end(session, user.username)
        else:
            st.caption(f"Ended by {session.get('closed_by') or '—'} at {session.get('closed_at') or '—'}.")

# =====================================================================================
# Manual entry
# =====================================================================================
with tab_manual:
    st.caption("For students the camera cannot mark (not in the trained model, no consent, camera problems).")
    open_sessions = service.open_sessions()
    if not open_sessions:
        empty_state("No running session", "Start one on Live Attendance or the Sessions tab.", ":material/event_busy:")
    else:
        by_id = {int(s["id"]): s for s in open_sessions}
        sid = st.selectbox("Session", list(by_id), format_func=lambda k: session_label(by_id[k]), key="manual_session")
        unmarked = {int(r["student_pk"]): r for r in sessions_repo.roster(sid) if r["attendance_id"] is None}
        if not unmarked:
            st.info("Everyone on this session's roster already has a mark. Use 'Correct a record' to change one.")
        else:
            with st.form("manual_entry"):
                pk = st.selectbox("Student", list(unmarked),
                                  format_func=lambda k: f"{unmarked[k]['student_id']} · {unmarked[k]['full_name']}")
                m_status = st.selectbox("Status", list(ATTENDANCE_STATUSES), format_func=status_text)
                notes = st.text_input("Notes (optional)", max_chars=200)
                ok = st.form_submit_button("Save entry", type="primary", icon=":material/save:")
            if ok:
                result = service.manual_entry(student_pk=pk, session_id=sid, status=m_status, operator=user.username,
                                              notes=notes)
                if result.ok:
                    st.session_state[FLASH_KEY] = ("success", result.message)
                    st.rerun()
                (st.info if result.duplicate else st.error)(result.message)

# =====================================================================================
# Corrections & logs
# =====================================================================================


@st.dialog("Delete old recognition logs")
def confirm_cleanup(days: int) -> None:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    st.write(f"Delete recognition log entries older than **{days} days**? Attendance records are not affected.")
    c1, c2 = st.columns(2)
    if c1.button("Delete", type="primary", key="dlg_logs_yes"):
        deleted = RecognitionLogRepository(db).delete_older_than(cutoff)
        st.session_state[FLASH_KEY] = ("success", f"Deleted {deleted} log entr{'y' if deleted == 1 else 'ies'}.")
        st.rerun()
    if c2.button("Cancel", key="dlg_logs_no"):
        st.rerun()


with tab_audit:
    st.markdown("##### Corrections")
    corrections = records_repo.corrections()
    if not corrections:
        st.caption("No corrections yet.")
    else:
        cdf = pd.DataFrame(corrections)
        st.dataframe(cdf[["changed_at", "attendance_id", "session_date", "session_name", "full_name", "previous_status",
                          "new_status", "reason", "changed_by"]], hide_index=True)
    st.markdown("##### Recognition events")
    st.caption("Only meaningful events are logged (marked, confirmed, duplicate, undone, rejected, uncertain photo) — "
               "never video frames or face images.")
    logs = RecognitionLogRepository(db).recent(200)
    if not logs:
        st.caption("No recognition events yet.")
    else:
        ldf = pd.DataFrame(logs)
        ldf["confidence"] = ldf["confidence"].map(lambda v: None if pd.isna(v) else round(float(v) * 100, 1))
        st.dataframe(ldf, hide_index=True,
                     column_config={"confidence": st.column_config.NumberColumn("Score", format="%.1f%%")})
        c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
        days = c1.number_input("Older than (days)", min_value=1, max_value=3650, value=30, step=1, key="log_days")
        if c2.button("Delete old log entries…", icon=":material/delete_sweep:", key="log_cleanup"):
            confirm_cleanup(int(days))
