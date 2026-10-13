"""Attendance sessions, face-recognition marking (automatic or confirmed), manual entries, corrections.

Rules:
* Attendance from the camera is written only after the SAME student was recognised in N
  consecutive frames (stability), and only for active, consenting students linked to the model.
* "Auto-mark" writes Present immediately after stability; the teacher can Undo it. With auto-mark
  off, the teacher clicks Confirm.
* Duplicates are impossible: SQLite's UNIQUE(student_id, session_id).
* Absences are created only when the teacher ends (finalises) a session.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any

from database.db import Database, DatabaseError
from database.repositories import (
    AttendanceRepository,
    DuplicateSessionError,
    InactiveStudentError,
    NotFoundError,
    RecognitionLogRepository,
    SessionClosedError,
    SessionRepository,
    StudentRepository,
)
from services.student_service import RecognitionEligibility, eligibility
from utils.validators import ValidationError, validate_session


class IdentityStatus(str, Enum):
    ELIGIBLE = "eligible"
    ALREADY_RECORDED = "already_recorded"
    UNMAPPED = "unmapped"
    INACTIVE = "inactive"
    NOT_CONSENTED = "not_consented"
    NO_SESSION = "no_session"
    SESSION_CLOSED = "session_closed"
    WRONG_CLASS = "wrong_class"


@dataclass(frozen=True)
class IdentityResolution:
    status: IdentityStatus
    student: dict[str, Any] | None
    message: str
    existing: dict[str, Any] | None = None

    @property
    def can_mark(self) -> bool:
        return self.status is IdentityStatus.ELIGIBLE

    @property
    def may_show_identity(self) -> bool:
        """Name / ID are shown only for an active, consenting, linked student."""
        return self.student is not None and self.status not in (
            IdentityStatus.UNMAPPED, IdentityStatus.INACTIVE, IdentityStatus.NOT_CONSENTED)


@dataclass(frozen=True)
class MarkResult:
    ok: bool
    duplicate: bool
    message: str
    attendance_id: int | None = None
    student: dict[str, Any] | None = None


_LOG_FOR = {
    IdentityStatus.UNMAPPED: "unmapped",
    IdentityStatus.INACTIVE: "inactive",
    IdentityStatus.NOT_CONSENTED: "not_consented",
    IdentityStatus.ALREADY_RECORDED: "duplicate",
    IdentityStatus.SESSION_CLOSED: "session_closed",
}


def suggested_status(session: dict[str, Any] | None, *, mark_late: bool, late_after_minutes: int,
                     now: datetime | None = None) -> str:
    """'present', or 'late' when late marking is enabled and we are past start + grace (today only)."""
    if not mark_late or not session or not session.get("start_time"):
        return "present"
    now = now or datetime.now()
    if session.get("session_date") != now.date().isoformat():
        return "present"
    try:
        start = datetime.combine(now.date(), time.fromisoformat(session["start_time"]))
    except ValueError:
        return "present"
    return "late" if now > start + timedelta(minutes=late_after_minutes) else "present"


class AttendanceService:
    def __init__(self, db: Database, *, log_events: bool = True) -> None:
        self.students = StudentRepository(db)
        self.sessions = SessionRepository(db)
        self.attendance = AttendanceRepository(db)
        self.logs = RecognitionLogRepository(db)
        self.log_events = log_events

    # ---- sessions ---------------------------------------------------------------------------

    def create_session(self, *, session_name: str, session_date: date | str, class_name: str, section: str | None,
                       start_time: time | str | None, end_time: time | str | None, operator: str) -> int:
        data = validate_session(session_name=session_name, session_date=session_date, class_name=class_name,
                                section=section, start_time=start_time, end_time=end_time)
        return self.sessions.create(data, created_by=operator)

    def start_quick_session(self, *, class_name: str, operator: str) -> int:
        """One-click session for today, starting now (e.g. 'TECH 405 class 09:05')."""
        now = datetime.now()
        base = f"{class_name} class {now:%H:%M}"
        for name in (base, f"{base}:{now:%S}"):
            try:
                return self.create_session(session_name=name, session_date=now.date(), class_name=class_name,
                                           section=None, start_time=now.strftime("%H:%M"), end_time=None,
                                           operator=operator)
            except DuplicateSessionError:
                continue
        raise DuplicateSessionError("A session was just created for this class. Select it from the list.")

    def close_session(self, session_id: int, *, operator: str) -> int:
        return self.sessions.close(session_id, closed_by=operator)

    def open_sessions(self) -> list[dict[str, Any]]:
        return self.sessions.list(status="open")

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        return self.sessions.get(session_id)

    # ---- recognition ------------------------------------------------------------------------

    def resolve_identity(self, label: str, session_id: int | None) -> IdentityResolution:
        """Map a stable prediction (model class) to a student and decide whether it may be marked."""
        student = self.students.get_by_model_label(label)
        if student is None:
            return IdentityResolution(IdentityStatus.UNMAPPED, None,
                                      "The recognised face is not linked to any registered student.")
        state = eligibility(student, {label})
        if state is RecognitionEligibility.INACTIVE:
            return IdentityResolution(IdentityStatus.INACTIVE, None, "The matched student is deactivated.")
        if state is RecognitionEligibility.NO_CONSENT:
            return IdentityResolution(IdentityStatus.NOT_CONSENTED, None,
                                      "Face-recognition consent is not recorded for the matched student — mark manually.")
        if session_id is None:
            return IdentityResolution(IdentityStatus.NO_SESSION, student, "Start an attendance session first.")
        session = self.sessions.get(session_id)
        if session is None:
            return IdentityResolution(IdentityStatus.NO_SESSION, student, "The selected session no longer exists.")
        if session["status"] != "open":
            return IdentityResolution(IdentityStatus.SESSION_CLOSED, student, "This session has been ended.")
        if student["class_name"].casefold() != session["class_name"].casefold() or (
                session.get("section") and (student.get("section") or "").casefold() != session["section"].casefold()):
            return IdentityResolution(IdentityStatus.WRONG_CLASS, student,
                                      f"{student['full_name']} is enrolled in {student['class_name']}, "
                                      f"not in this session's class ({session['class_name']}).")
        existing = self.attendance.get_for(int(student["id"]), session_id)
        if existing is not None:
            return IdentityResolution(IdentityStatus.ALREADY_RECORDED, student,
                                      f"{student['full_name']} is already marked {existing['status']} "
                                      f"(at {str(existing['recorded_at'])[11:16]}).", existing)
        return IdentityResolution(IdentityStatus.ELIGIBLE, student, "Ready to mark Present.")

    def _log(self, status: str, label: str | None, confidence: float | None, student_pk: int | None,
             session_id: int | None, source: str | None) -> None:
        if not self.log_events:
            return
        try:
            self.logs.add(recognition_status=status, predicted_label=label, confidence=confidence,
                          student_pk=student_pk, session_id=session_id, source=source)
        except DatabaseError:
            pass  # logging never blocks attendance

    def mark_recognised(self, *, label: str, confidence: float, session_id: int, status: str, operator: str,
                        source: str, automatic: bool) -> MarkResult:
        """Write a face-recognition attendance record (automatic, or after the teacher's confirmation)."""
        if status not in ("present", "late"):
            return MarkResult(False, False, "Face recognition can only record Present or Late.")
        resolution = self.resolve_identity(label, session_id)
        student_pk = int(resolution.student["id"]) if resolution.student else None
        if not resolution.can_mark:
            self._log(_LOG_FOR.get(resolution.status, "error"), label, confidence, student_pk, session_id, source)
            return MarkResult(False, resolution.status is IdentityStatus.ALREADY_RECORDED, resolution.message,
                              student=resolution.student if resolution.may_show_identity else None)
        assert resolution.student is not None
        try:
            outcome = self.attendance.record(
                student_pk=int(resolution.student["id"]), session_id=session_id, status=status,
                method="face_recognition",
                confirmed_by=f"{operator} (auto)" if automatic else operator,
                confidence=round(min(max(float(confidence), 0.0), 1.0), 4),
                notes=("Auto-marked after stable live recognition" if automatic
                       else f"Confirmed by teacher after {source.replace('_', ' ')} recognition"),
            )
        except (SessionClosedError, InactiveStudentError, NotFoundError) as exc:
            self._log("error", label, confidence, student_pk, session_id, source)
            return MarkResult(False, False, str(exc))
        name = resolution.student["full_name"]
        if not outcome.inserted:
            self._log("duplicate", label, confidence, student_pk, session_id, source)
            return MarkResult(False, True, f"{name} is already marked ({outcome.existing_status}).",
                              outcome.attendance_id, resolution.student)
        self._log("auto_marked" if automatic else "confirmed", label, confidence, student_pk, session_id, source)
        return MarkResult(True, False, f"{name} marked {status.capitalize()}.", outcome.attendance_id, resolution.student)

    def undo_mark(self, attendance_id: int, *, source: str = "live_video") -> MarkResult:
        """Teacher undo of a face-recognition mark in an open session (e.g. a wrong match)."""
        try:
            row = self.attendance.delete_face_record(attendance_id)
        except DatabaseError as exc:
            return MarkResult(False, False, str(exc))
        self._log("undone", row.get("model_class_label"), row.get("confidence"), int(row["student_id"]),
                  int(row["session_id"]), source)
        return MarkResult(True, False, "Mark removed. The student can be recognised again.")

    def reject(self, *, label: str | None, confidence: float | None, session_id: int | None, source: str) -> None:
        student = self.students.get_by_model_label(label) if label else None
        self._log("rejected_by_operator", label, confidence, int(student["id"]) if student else None, session_id, source)

    def log_still_outcome(self, *, status: str, label: str | None, confidence: float | None,
                          session_id: int | None) -> None:
        self._log(status, label, confidence, None, session_id, "still_image")

    # ---- manual -----------------------------------------------------------------------------

    def manual_entry(self, *, student_pk: int, session_id: int, status: str, operator: str,
                     notes: str | None) -> MarkResult:
        try:
            outcome = self.attendance.record(student_pk=student_pk, session_id=session_id, status=status,
                                             method="manual", confirmed_by=operator, confidence=None,
                                             notes=(notes or "").strip() or "Manual entry")
        except (SessionClosedError, InactiveStudentError, NotFoundError, ValueError) as exc:
            return MarkResult(False, False, str(exc))
        if not outcome.inserted:
            return MarkResult(False, True, f"A record already exists ({outcome.existing_status}). "
                                           "Use 'Correct a record' to change it.", outcome.attendance_id)
        return MarkResult(True, False, "Manual attendance entry saved.", outcome.attendance_id)

    def correct(self, attendance_id: int, *, new_status: str, reason: str, operator: str) -> bool:
        if len((reason or "").strip()) < 3:
            raise ValidationError(["Please give a reason for the correction (at least 3 characters)."])
        return self.attendance.correct_status(attendance_id, new_status=new_status, reason=reason, changed_by=operator)
