"""Student portal: the signed-in student's own attendance (nobody else's)."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.reporting_service import ReportingService
from services.student_service import ELIGIBILITY_TEXT, StudentService, eligibility
from ui.auth_ui import require_student
from ui.components import avatar, clock, download_csv, empty_state, format_rate, metric_card, page_header, status_text
from ui.state import artifact_status, get_paths, require_database
from ui.theme import STATUS_COLORS, STATUS_LABELS, status_badge, style_figure

user = require_student()
db = require_database()
students = StudentService(db, get_paths().dataset_dir)
student = students.get(int(user.student_pk))
if student is None:
    st.error("Your student record no longer exists. Please contact the teacher.")
    st.stop()

page_header(f"Hello, {student['full_name']}", "Your attendance in this course.", kicker="TECH 405 · Student portal")
overview = ReportingService(db).student_overview(int(student["id"]))
counts = overview["counts"]

left, right = st.columns([1, 3], gap="large")
with left:
    with st.container(border=True):
        avatar(students.photo(student), student["full_name"])
        st.markdown(f'<div class="sas-name">{html.escape(student["full_name"])}</div>'
                    f'<div class="sas-muted">ID {html.escape(student["student_id"])} · {html.escape(student["class_name"])}'
                    "</div>", unsafe_allow_html=True)
        state = eligibility(student, artifact_status().label_set)
        st.caption(f":material/videocam: {ELIGIBILITY_TEXT[state]}")
with right:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Attendance rate", format_rate(overview["rate"]),
                    help="(present + late) / (present + late + absent) over ended sessions.")
    with k2:
        metric_card("Sessions attended", overview["attended_total"])
    with k3:
        metric_card("Absences", counts["absent"])
    with k4:
        metric_card("Last attended", str(overview["last_seen"])[:10] if overview["last_seen"] else "—")

    history: pd.DataFrame = overview["history"]
    if history.empty:
        empty_state("No attendance yet", "Your attendance appears here after the teacher's camera marks you or a "
                    "session ends.", ":material/event_note:")
    else:
        st.markdown("##### Latest")
        latest = history.iloc[0]
        st.markdown(f"{latest['session_date']:%d %b %Y} · {html.escape(str(latest['session_name']))} — "
                    + status_badge(latest["status"], clock(latest["recorded_at"])), unsafe_allow_html=True)

if not history.empty:
    st.subheader("Attendance by month")
    monthly = history.assign(month=history["session_date"].dt.strftime("%Y-%m"))
    fig = go.Figure()
    for status in ("present", "late", "absent", "excused"):
        per_month = monthly[monthly["status"] == status].groupby("month").size()
        months = sorted(monthly["month"].unique())
        fig.add_bar(x=months, y=[int(per_month.get(m, 0)) for m in months], name=STATUS_LABELS[status],
                    marker_color=STATUS_COLORS[status], hovertemplate=f"{STATUS_LABELS[status]}: %{{y}}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category")
    st.plotly_chart(style_figure(fig, height=300), key="student_monthly")

    st.subheader("History")
    view = history.copy()
    view["date"] = view["session_date"].dt.date
    view["time"] = view["recorded_at"].map(clock)
    view["status"] = view["status"].map(status_text)
    view["source"] = view["method"].map({"face_recognition": "Camera", "manual": "Teacher (manual)",
                                         "session_finalize": "Session ended"}).fillna(view["method"])
    view["session"] = view["session_status"].map({"open": "Running", "closed": "Ended"})
    st.dataframe(view[["date", "time", "session_name", "status", "source", "session"]], hide_index=True,
                 column_config={"date": "Date", "time": "Time", "session_name": "Session", "status": "Status",
                                "source": "Marked by", "session": "Session state"})
    download_csv(view[["date", "time", "session_name", "status", "source"]], f"my_attendance_{student['student_id']}.csv",
                 "Download my attendance (CSV)", key="student_csv")

st.caption("If something looks wrong, ask your teacher — every correction is recorded in an audit trail.")
