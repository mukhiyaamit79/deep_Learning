"""Teacher dashboard: today's class at a glance, trends and system status."""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database.repositories import AttendanceRepository, SessionRepository, table_counts
from services.reporting_service import ReportFilters, ReportingService
from services.settings_service import load_settings
from services.student_service import StudentService
from ui.auth_ui import require_teacher
from ui.components import clock, empty_state, format_rate, metric_card, page_header, session_label, status_text, student_card
from ui.state import MODEL_LOAD_INFO, artifact_status, get_paths, model_loaded, require_database
from ui.theme import STATUS_COLORS, STATUS_LABELS, categorical, style_figure

user = require_teacher()
today = date.today()
page_header(f"Welcome, {user.display_name}", f"{today:%A, %d %B %Y}", kicker="TECH 405 · Teacher dashboard")

db = require_database()
paths = get_paths()
settings = load_settings(db)
reporting = ReportingService(db)
students = StudentService(db, paths.dataset_dir)
overview = reporting.day_overview(today)

# ---- KPIs -------------------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    metric_card("Enrolled students", students.count_active(), help="Active student records.")
with k2:
    metric_card("Present today", overview["attended"], help="Present + late marks in today's sessions.")
with k3:
    if overview["open_count"]:
        metric_card("Not yet marked", overview["not_yet_marked"],
                    help="Students in today's running sessions without a mark. They become absent only when you end the session.")
    else:
        metric_card("Absent today", overview["absent"] if overview["closed_count"] else "—",
                    help="Counted for ended sessions only.")
with k4:
    metric_card("Attendance rate (30 days)", format_rate(reporting.overall_rate(30, today)),
                help="(present + late) / (present + late + absent) over ended sessions in the last 30 days.")
with k5:
    metric_card("Sessions today", overview["session_count"],
                help=f"{overview['open_count']} running · {overview['closed_count']} ended")

# ---- Today's class --------------------------------------------------------------------------------
st.subheader("Today's class")
sessions_today = overview["sessions"]
if not sessions_today:
    empty_state("No attendance session today",
                "Open Live Attendance and press “Start today's session” — the camera will then mark students "
                "present automatically.", ":material/event_busy:")
    st.page_link("app_pages/live_attendance.py", label="Go to Live Attendance", icon=":material/videocam:")
else:
    by_id = {int(s["id"]): s for s in sessions_today}
    ids = list(by_id)
    default = next((sid for sid in ids if by_id[sid]["status"] == "open"), ids[0])
    chosen = st.selectbox("Session", ids, index=ids.index(default), format_func=lambda k: session_label(by_id[k]),
                          key="dash_session")
    session = by_id[chosen]
    roster = SessionRepository(db).roster(chosen)
    marked = int(session["attended_count"])
    total = max(int(session["roster_count"]), 1)
    st.progress(min(marked / total, 1.0), text=f"{marked} of {session['roster_count']} students present"
                + (" · session running" if session["status"] == "open" else " · session ended"))
    if not roster:
        st.caption("No active students belong to this session's class.")
    per_row = 5
    for start in range(0, len(roster), per_row):
        cols = st.columns(per_row)
        for col, row in zip(cols, roster[start : start + per_row]):
            with col:
                status = row["status"] if row["status"] else "not marked"
                detail = None
                if status in ("present", "late"):
                    detail = clock(row["recorded_at"])
                    if row["method"] == "face_recognition" and row["confidence"] is not None:
                        detail += f" · {float(row['confidence']):.0%}"
                    elif row["method"] == "manual":
                        detail += " · manual"
                student_card(row, students.photo(row), status, detail)
    st.page_link("app_pages/live_attendance.py", label="Open the live camera for this session", icon=":material/videocam:")

# ---- Charts ---------------------------------------------------------------------------------------
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Attendance trend")
    trend = reporting.trend(30, today)
    if trend.empty:
        empty_state("No ended sessions in the last 30 days", "Trends use ended sessions only, so a running session "
                    "never looks like low attendance.", ":material/show_chart:")
    else:
        fig = go.Figure(go.Scatter(
            x=trend["date"], y=trend["attendance_rate"], mode="lines+markers", line=dict(color=categorical(0)),
            customdata=trend[["present", "late", "absent"]].to_numpy(),
            hovertemplate="%{x|%d %b}<br>Rate %{y:.0%}<br>Present %{customdata[0]} · Late %{customdata[1]} · "
                          "Absent %{customdata[2]}<extra></extra>"))
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        st.plotly_chart(style_figure(fig, height=300, legend=False), key="dash_trend")

with right:
    st.subheader("Attendance rate by student")
    by_student = reporting.by_student(ReportFilters(today - timedelta(days=89), today, None, True))
    if by_student.empty:
        empty_state("No ended sessions yet", "End a session on Live Attendance to record absences and see rates.",
                    ":material/bar_chart:")
    else:
        ordered = by_student.sort_values("attendance_rate")
        fig = go.Figure(go.Bar(
            x=ordered["attendance_rate"], y=ordered["full_name"], orientation="h", marker_color=categorical(0),
            text=[format_rate(v) for v in ordered["attendance_rate"]], textposition="outside",
            customdata=ordered[["present", "late", "absent"]].to_numpy(),
            hovertemplate="%{y}<br>Rate %{x:.0%}<br>Present %{customdata[0]} · Late %{customdata[1]} · "
                          "Absent %{customdata[2]}<extra></extra>"))
        fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
        fig.update_yaxes(type="category")
        st.plotly_chart(style_figure(fig, height=max(260, 34 * len(ordered) + 70), legend=False), key="dash_by_student")
        st.caption("Last 90 days, ended sessions only.")

# ---- Status mix today + recent activity ---------------------------------------------------------------
c1, c2 = st.columns([1, 2], gap="large")
with c1:
    st.subheader("Today by status")
    statuses = ["present", "late", "absent", "excused", "not marked"]
    counts = [sum(int(s[f"{k}_count"]) for s in sessions_today) for k in statuses[:4]] + [overview["not_yet_marked"]]
    if sum(counts) == 0:
        st.caption("Nothing recorded today yet.")
    else:
        fig = go.Figure(go.Bar(x=[STATUS_LABELS[s] for s in statuses], y=counts,
                               marker_color=[STATUS_COLORS[s] for s in statuses], text=counts, textposition="outside",
                               hovertemplate="%{x}: %{y}<extra></extra>"))
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(style_figure(fig, height=280, legend=False), key="dash_status")
with c2:
    st.subheader("Recent activity")
    recent = AttendanceRepository(db).recent(int(settings["display.recent_records_limit"]))
    if not recent:
        st.caption("No attendance recorded yet.")
    else:
        df = pd.DataFrame(recent)
        df["status"] = df["status"].map(status_text)
        df["time"] = df["recorded_at"].map(clock)
        df["confidence"] = df["confidence"].map(lambda v: None if pd.isna(v) else round(float(v) * 100, 1))
        st.dataframe(df[["session_date", "time", "full_name", "status", "method", "confidence", "session_name"]],
                     hide_index=True,
                     column_config={"session_date": "Date", "time": "Time", "full_name": "Student", "status": "Status",
                                    "method": "Source", "session_name": "Session",
                                    "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%")})

# ---- System status ---------------------------------------------------------------------------------
with st.expander("System status", icon=":material/monitor_heart:"):
    status = artifact_status()
    s1, s2, s3 = st.columns(3)
    with s1:
        counts_db = table_counts(db)
        st.markdown(":green[:material/check_circle:] **SQLite database ready**")
        st.caption(f"{counts_db['students']} students · {counts_db['users']} accounts · "
                   f"{counts_db['attendance_sessions']} sessions · {counts_db['attendance']} attendance rows")
    with s2:
        if status.ready:
            st.markdown(f":green[:material/check_circle:] **Trained CNN found** ({len(status.class_names or ())} students)")
            if model_loaded(status):
                st.caption(f"Loaded on {MODEL_LOAD_INFO['device']} · PyTorch {MODEL_LOAD_INFO['torch_version']} · "
                           f"{MODEL_LOAD_INFO['parameters']:,} parameters")
            else:
                st.caption("Loads when Live Attendance opens.")
        else:
            st.markdown(":orange[:material/warning:] **Model files missing**")
            for err in status.errors:
                st.caption(f"• {err}")
    with s3:
        ok = importlib.util.find_spec("streamlit_webrtc") is not None
        st.markdown(":green[:material/check_circle:] **Live camera component installed**" if ok
                    else ":orange[:material/warning:] **streamlit-webrtc missing** — still-photo mode only")
        st.caption("Results from the notebook: " + ("found" if paths.comparison_csv.is_file() else "not found"))
