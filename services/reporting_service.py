"""Attendance statistics computed from the real SQLite records.

Attendance rate (used everywhere):
    (present + late) / (present + late + absent)
* Only sessions that actually took place are counted; days without a session are never absences.
* By default only ENDED (finalised) sessions are included. While a session is running, students
  without a record are "not yet marked", not absent.
* 'excused' records are excluded from the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from database.db import Database
from database.repositories import AttendanceRepository, SessionRepository

STATUS_COLUMNS: tuple[str, ...] = ("present", "late", "absent", "excused")
REPORT_COLUMNS: tuple[str, ...] = ("sessions", "present", "late", "absent", "excused", "records", "attendance_rate")
ALL_TIME_START = date(2000, 1, 1)


def attendance_rate(attended: float, absent: float) -> float:
    denominator = attended + absent
    return float(attended) / float(denominator) if denominator > 0 else float("nan")


def _empty(group_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*group_cols, *REPORT_COLUMNS])


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return _empty(group_cols)
    work = df.copy()
    for status in STATUS_COLUMNS:
        work[status] = (work["status"] == status).astype(int)
    grouped = work.groupby(group_cols, dropna=False).agg(
        sessions=("session_id", "nunique"), present=("present", "sum"), late=("late", "sum"),
        absent=("absent", "sum"), excused=("excused", "sum"), records=("status", "size"),
    ).reset_index()
    attended = grouped["present"] + grouped["late"]
    denominator = attended + grouped["absent"]
    grouped["attendance_rate"] = np.where(denominator > 0, attended / denominator.where(denominator > 0, 1), np.nan)
    return grouped


@dataclass(frozen=True)
class ReportFilters:
    date_from: date
    date_to: date
    class_name: str | None = None
    finalized_only: bool = True
    student_pk: int | None = None


_ROW_COLUMNS = ["session_id", "session_date", "session_name", "session_class", "session_section", "session_status",
                "start_time", "student_pk", "student_id", "full_name", "class_name", "section", "status", "method",
                "recorded_at", "confidence"]


class ReportingService:
    def __init__(self, db: Database) -> None:
        self.attendance = AttendanceRepository(db)
        self.sessions = SessionRepository(db)

    def rows(self, f: ReportFilters) -> pd.DataFrame:
        data = self.attendance.report_rows(date_from=f.date_from.isoformat(), date_to=f.date_to.isoformat(),
                                           class_name=f.class_name, finalized_only=f.finalized_only,
                                           student_pk=f.student_pk)
        if not data:
            return pd.DataFrame(columns=_ROW_COLUMNS)
        df = pd.DataFrame(data)
        df["session_date"] = pd.to_datetime(df["session_date"])
        return df[_ROW_COLUMNS]

    def daily(self, f: ReportFilters) -> pd.DataFrame:
        df = self.rows(f)
        if df.empty:
            return _empty(["date"])
        df["date"] = df["session_date"].dt.date
        return aggregate(df, ["date"]).sort_values("date")

    def weekly(self, f: ReportFilters) -> pd.DataFrame:
        df = self.rows(f)
        if df.empty:
            return _empty(["week_start"])
        df["week_start"] = (df["session_date"] - pd.to_timedelta(df["session_date"].dt.weekday, unit="D")).dt.date
        return aggregate(df, ["week_start"]).sort_values("week_start")

    def monthly(self, f: ReportFilters) -> pd.DataFrame:
        df = self.rows(f)
        if df.empty:
            return _empty(["month"])
        df["month"] = df["session_date"].dt.strftime("%Y-%m")
        return aggregate(df, ["month"]).sort_values("month")

    def by_student(self, f: ReportFilters) -> pd.DataFrame:
        df = self.rows(f)
        if df.empty:
            return _empty(["student_id", "full_name", "class_name", "section"])
        df["section"] = df["section"].fillna("")
        return aggregate(df, ["student_id", "full_name", "class_name", "section"]).sort_values(["class_name", "full_name"])

    def by_class(self, f: ReportFilters) -> pd.DataFrame:
        df = self.rows(f)
        if df.empty:
            return _empty(["class_name"])
        return aggregate(df, ["session_class"]).rename(columns={"session_class": "class_name"}).sort_values("class_name")

    def by_session(self, f: ReportFilters) -> pd.DataFrame:
        df = self.rows(f)
        if df.empty:
            return _empty(["session_date", "session_name", "session_class", "session_status"])
        df["session_date"] = df["session_date"].dt.date
        return aggregate(df, ["session_id", "session_date", "session_name", "session_class", "session_status"]
                         ).sort_values(["session_date", "session_name"])

    def status_totals(self, f: ReportFilters) -> dict[str, int]:
        df = self.rows(f)
        return {s: int((df["status"] == s).sum()) if not df.empty else 0 for s in STATUS_COLUMNS}

    # ---- dashboards ---------------------------------------------------------------------------

    def day_overview(self, day: date) -> dict[str, Any]:
        sessions = self.sessions.list(date_from=day.isoformat(), date_to=day.isoformat())
        closed = [s for s in sessions if s["status"] == "closed"]
        open_ = [s for s in sessions if s["status"] == "open"]
        absent_closed = sum(int(s["absent_count"]) for s in closed)
        attended_closed = sum(int(s["attended_count"]) for s in closed)
        return {
            "sessions": sessions,
            "session_count": len(sessions),
            "open_count": len(open_),
            "closed_count": len(closed),
            "attended": sum(int(s["attended_count"]) for s in sessions),
            "absent": absent_closed,
            "not_yet_marked": sum(int(s["not_marked_count"]) for s in open_),
            "rate_closed": attendance_rate(attended_closed, absent_closed),
        }

    def trend(self, days: int = 30, today: date | None = None) -> pd.DataFrame:
        today = today or date.today()
        return self.daily(ReportFilters(today - timedelta(days=days - 1), today, None, True))

    def overall_rate(self, days: int = 30, today: date | None = None) -> float:
        today = today or date.today()
        totals = self.status_totals(ReportFilters(today - timedelta(days=days - 1), today, None, True))
        return attendance_rate(totals["present"] + totals["late"], totals["absent"])

    def student_overview(self, student_pk: int) -> dict[str, Any]:
        """All-time summary for one student (student portal)."""
        everything = self.rows(ReportFilters(ALL_TIME_START, date.today() + timedelta(days=1), None, False, student_pk))
        finalised = everything[everything["session_status"] == "closed"] if not everything.empty else everything
        counts = {s: int((finalised["status"] == s).sum()) if not finalised.empty else 0 for s in STATUS_COLUMNS}
        attended_all = everything[everything["status"].isin(["present", "late"])] if not everything.empty else everything
        last_seen = None
        if not attended_all.empty:
            last_seen = str(attended_all.sort_values("recorded_at").iloc[-1]["recorded_at"])
        return {
            "history": everything.sort_values(["session_date", "recorded_at"], ascending=False)
            if not everything.empty else everything,
            "counts": counts,
            "rate": attendance_rate(counts["present"] + counts["late"], counts["absent"]),
            "sessions_recorded": int(len(everything)),
            "attended_total": int(len(attended_all)),
            "last_seen": last_seen,
        }
