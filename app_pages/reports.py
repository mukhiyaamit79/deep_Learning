"""Attendance reports: daily, weekly, monthly, per student, per class and per session."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.reporting_service import ReportFilters, ReportingService, attendance_rate
from services.student_service import StudentService
from ui.auth_ui import require_teacher
from ui.components import download_csv, empty_state, format_rate, metric_card, page_header
from ui.state import get_paths, require_database
from ui.theme import STATUS_COLORS, STATUS_LABELS, categorical, style_figure

require_teacher()
page_header("Reports", "Summaries computed from the attendance stored in SQLite, downloadable as CSV.")

db = require_database()
reporting = ReportingService(db)
students = StudentService(db, get_paths().dataset_dir)

with st.expander("How the attendance rate is calculated", icon=":material/calculate:"):
    st.markdown(
        "**Rate = (present + late) ÷ (present + late + absent).** Only sessions you actually ran are counted — days "
        "without a session are never absences. By default only **ended** sessions are used: while a session is "
        "running, students without a mark are *not yet marked*. Ending a session records the remaining students of "
        "that class as absent. *Excused* is left out of the denominator."
    )

today = date.today()
f1, f2, f3 = st.columns([1.4, 1, 1], vertical_alignment="bottom")
with f1:
    value = st.date_input("Session dates", value=(today - timedelta(days=60), today), format="YYYY-MM-DD", key="rep_dates")
    d_from, d_to = ((value[0], value[-1]) if value else (today, today)) if isinstance(value, (tuple, list)) else (value, value)
class_choice = f2.selectbox("Class", ["All classes", *students.class_names()], key="rep_class")
finalized_only = f3.toggle("Ended sessions only", value=True, help="Recommended.")
filters = ReportFilters(d_from, d_to, None if class_choice == "All classes" else class_choice, finalized_only)

totals = reporting.status_totals(filters)
cols = st.columns(5)
for col, (label, val) in zip(cols, [("Present", totals["present"]), ("Late", totals["late"]), ("Absent", totals["absent"]),
                                    ("Excused", totals["excused"]),
                                    ("Attendance rate", format_rate(attendance_rate(totals["present"] + totals["late"],
                                                                                    totals["absent"])))]):
    with col:
        metric_card(label, val)

if sum(totals.values()) == 0:
    empty_state("No attendance data for these filters",
                "Reports use ended sessions by default — end a session on Live Attendance, or switch off "
                "'Ended sessions only'.", ":material/query_stats:")
    st.stop()


def status_stack(df: pd.DataFrame, x: str, x_title: str, key: str) -> None:
    fig = go.Figure()
    for status in ("present", "late", "absent", "excused"):
        fig.add_bar(x=df[x], y=df[status], name=STATUS_LABELS[status], marker_color=STATUS_COLORS[status],
                    hovertemplate=f"{STATUS_LABELS[status]}: %{{y}}<extra></extra>")
    fig.update_layout(barmode="stack", title=dict(text="Marks by status", font_size=14), hovermode="x unified")
    fig.update_xaxes(title_text=x_title, type="category")
    st.plotly_chart(style_figure(fig), key=f"{key}_stack")


def rate_line(df: pd.DataFrame, x: str, x_title: str, key: str) -> None:
    fig = go.Figure(go.Scatter(x=df[x], y=df["attendance_rate"], mode="lines+markers", line=dict(color=categorical(0)),
                               customdata=df[["sessions", "records"]].to_numpy(),
                               hovertemplate="%{x}<br>Rate %{y:.0%}<br>Sessions %{customdata[0]} · Marks %{customdata[1]}"
                                             "<extra></extra>"))
    fig.update_layout(title=dict(text="Attendance rate", font_size=14))
    fig.update_xaxes(title_text=x_title, type="category")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    st.plotly_chart(style_figure(fig, legend=False), key=f"{key}_rate")


def rate_bars(df: pd.DataFrame, label_col: str, title: str, key: str) -> None:
    ordered = df.sort_values("attendance_rate", na_position="first")
    fig = go.Figure(go.Bar(x=ordered["attendance_rate"], y=ordered[label_col], orientation="h", marker_color=categorical(0),
                           text=[format_rate(v) for v in ordered["attendance_rate"]], textposition="outside",
                           customdata=ordered[["present", "late", "absent", "sessions"]].to_numpy(),
                           hovertemplate="%{y}<br>Rate %{x:.0%}<br>Present %{customdata[0]} · Late %{customdata[1]} · "
                                         "Absent %{customdata[2]} · Sessions %{customdata[3]}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font_size=14))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
    fig.update_yaxes(type="category")
    st.plotly_chart(style_figure(fig, height=max(260, 32 * len(ordered) + 80), legend=False), key=f"{key}_bars")


def table(df: pd.DataFrame, filename: str, key: str) -> None:
    view = df.copy()
    view["attendance_rate"] = view["attendance_rate"].map(lambda v: None if pd.isna(v) else round(float(v) * 100, 1))
    st.dataframe(view, hide_index=True,
                 column_config={"attendance_rate": st.column_config.NumberColumn("Attendance rate", format="%.1f%%")})
    download_csv(df, filename, "Download report (CSV)", key=f"{key}_csv")


stamp = f"{filters.date_from}_{filters.date_to}"
t1, t2, t3, t4, t5, t6 = st.tabs(["Daily", "Weekly", "Monthly", "Per student", "Per class", "Per session"])
with t1:
    daily = reporting.daily(filters)
    daily["date"] = daily["date"].astype(str)
    a, b = st.columns(2)
    with a:
        status_stack(daily, "date", "Date", "daily")
    with b:
        rate_line(daily, "date", "Date", "daily")
    table(daily, f"daily_attendance_{stamp}.csv", "daily")
with t2:
    weekly = reporting.weekly(filters)
    weekly["week_start"] = weekly["week_start"].astype(str)
    a, b = st.columns(2)
    with a:
        status_stack(weekly, "week_start", "Week starting (Monday)", "weekly")
    with b:
        rate_line(weekly, "week_start", "Week starting (Monday)", "weekly")
    table(weekly, f"weekly_attendance_{stamp}.csv", "weekly")
with t3:
    monthly = reporting.monthly(filters)
    a, b = st.columns(2)
    with a:
        status_stack(monthly, "month", "Month", "monthly")
    with b:
        rate_line(monthly, "month", "Month", "monthly")
    table(monthly, f"monthly_attendance_{stamp}.csv", "monthly")
with t4:
    per_student = reporting.by_student(filters)
    per_student["label"] = per_student["full_name"].astype(str) + " (" + per_student["student_id"].astype(str) + ")"
    rate_bars(per_student, "label", "Attendance rate per student", "student")
    table(per_student.drop(columns=["label"]), f"student_attendance_{stamp}.csv", "student")
with t5:
    per_class = reporting.by_class(filters)
    rate_bars(per_class, "class_name", "Attendance rate per class", "class")
    table(per_class, f"class_attendance_{stamp}.csv", "class")
with t6:
    table(reporting.by_session(filters), f"session_attendance_{stamp}.csv", "session")
