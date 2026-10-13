"""Repositories: every SQL statement of the web app lives in this module.

Pages call services; services call these classes. All queries are parameterised. Integrity rules
(unique student IDs and usernames, one attendance row per student and session, valid confidence
range, ...) are enforced by the schema; the Python checks only give friendlier messages.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from database.db import Database, DatabaseError
from utils.validators import SessionInput, StudentInput, now_iso

ATTENDANCE_STATUSES: tuple[str, ...] = ("present", "late", "absent", "excused")
ATTENDED_STATUSES: tuple[str, ...] = ("present", "late")
ATTENDANCE_METHODS: tuple[str, ...] = ("face_recognition", "manual", "session_finalize")


class NotFoundError(DatabaseError):
    pass


class DuplicateStudentError(DatabaseError):
    pass


class ModelLabelConflictError(DatabaseError):
    pass


class DuplicateSessionError(DatabaseError):
    pass


class DuplicateUserError(DatabaseError):
    pass


class SessionClosedError(DatabaseError):
    pass


class InactiveStudentError(DatabaseError):
    pass


def _like(term: str | None) -> str | None:
    """LIKE pattern for free-text search, escaping wildcards (escape char '!')."""
    if term is None or not term.strip():
        return None
    escaped = term.strip().replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return f"%{escaped}%"


def _in_clause(prefix: str, values: Iterable[Any]) -> tuple[str, dict[str, Any]]:
    params = {f"{prefix}{i}": value for i, value in enumerate(values)}
    return "(" + (", ".join(f":{name}" for name in params) or "NULL") + ")", params


# Students expected at a session (denominator for absences). Needs aliases st and se.
# Active students of the session's class (and section, if set) registered on/before the session date.
ROSTER_CONDITIONS = """
    st.is_active = 1
    AND st.class_name = se.class_name COLLATE NOCASE
    AND (se.section IS NULL OR COALESCE(st.section, '') = se.section COLLATE NOCASE)
    AND substr(st.created_at, 1, 10) <= se.session_date
"""

# ======================================================================================
# Students
# ======================================================================================

_STUDENT_COLUMNS = """
    id, student_id, full_name, class_name, section, email, model_class_label, photo_path,
    is_active, consent_recorded_at, created_at, updated_at
"""


class StudentRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _integrity(exc: sqlite3.IntegrityError, data: StudentInput) -> DatabaseError:
        message = str(exc)
        if "students.student_id" in message:
            return DuplicateStudentError(f"Student ID '{data.student_id}' is already registered.")
        if "model_class_label" in message:
            return ModelLabelConflictError(f"Model class '{data.model_class_label}' is already linked to another student.")
        return DatabaseError(f"The student record violates a database rule: {message}")

    def create(self, data: StudentInput, *, consent: bool, photo_path: str | None = None,
               conn: sqlite3.Connection | None = None) -> int:
        now = now_iso()
        params = {
            "student_id": data.student_id, "full_name": data.full_name, "class_name": data.class_name,
            "section": data.section, "email": data.email, "model_class_label": data.model_class_label,
            "photo_path": photo_path, "consent": now if consent else None, "now": now,
        }
        sql = """
            INSERT INTO students (student_id, full_name, class_name, section, email, model_class_label,
                                  photo_path, is_active, consent_recorded_at, created_at, updated_at)
            VALUES (:student_id, :full_name, :class_name, :section, :email, :model_class_label,
                    :photo_path, 1, :consent, :now, :now)
        """
        try:
            if conn is not None:  # caller-managed transaction (e.g. student + login account together)
                return int(conn.execute(sql, params).lastrowid)
            with self.db.transaction() as tx:
                return int(tx.execute(sql, params).lastrowid)
        except sqlite3.IntegrityError as exc:
            raise self._integrity(exc, data) from exc

    def update(self, pk: int, data: StudentInput) -> None:
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE students
                       SET student_id = :student_id, full_name = :full_name, class_name = :class_name,
                           section = :section, email = :email, model_class_label = :model_class_label,
                           updated_at = :now
                     WHERE id = :pk
                    """,
                    {"student_id": data.student_id, "full_name": data.full_name, "class_name": data.class_name,
                     "section": data.section, "email": data.email, "model_class_label": data.model_class_label,
                     "now": now_iso(), "pk": pk},
                )
                if cursor.rowcount == 0:
                    raise NotFoundError("Student not found.")
        except sqlite3.IntegrityError as exc:
            raise self._integrity(exc, data) from exc

    def set_model_label(self, pk: int, label: str | None) -> None:
        try:
            with self.db.transaction() as conn:
                conn.execute("UPDATE students SET model_class_label = ?, updated_at = ? WHERE id = ?",
                             (label, now_iso(), pk))
        except sqlite3.IntegrityError as exc:
            raise ModelLabelConflictError(f"Model class '{label}' is already linked to another student.") from exc

    def set_photo(self, pk: int, photo_path: str | None) -> None:
        with self.db.transaction() as conn:
            conn.execute("UPDATE students SET photo_path = ?, updated_at = ? WHERE id = ?", (photo_path, now_iso(), pk))

    def set_active(self, pk: int, active: bool) -> None:
        with self.db.transaction() as conn:
            if conn.execute("UPDATE students SET is_active = ?, updated_at = ? WHERE id = ?",
                            (1 if active else 0, now_iso(), pk)).rowcount == 0:
                raise NotFoundError("Student not found.")

    def set_consent(self, pk: int, consent: bool) -> None:
        now = now_iso()
        with self.db.transaction() as conn:
            if conn.execute("UPDATE students SET consent_recorded_at = ?, updated_at = ? WHERE id = ?",
                            (now if consent else None, now, pk)).rowcount == 0:
                raise NotFoundError("Student not found.")

    def get(self, pk: int) -> dict[str, Any] | None:
        return self.db.query_one(f"SELECT {_STUDENT_COLUMNS} FROM students WHERE id = ?", (pk,))

    def get_by_student_id(self, student_id: str) -> dict[str, Any] | None:
        return self.db.query_one(f"SELECT {_STUDENT_COLUMNS} FROM students WHERE student_id = ? COLLATE NOCASE",
                                 (student_id,))

    def get_by_model_label(self, label: str) -> dict[str, Any] | None:
        return self.db.query_one(f"SELECT {_STUDENT_COLUMNS} FROM students WHERE model_class_label = ?", (label,))

    def list(self, *, search: str | None = None, class_name: str | None = None,
             active: bool | None = None) -> list[dict[str, Any]]:
        return self.db.query_all(
            f"""
            SELECT {_STUDENT_COLUMNS}
              FROM students
             WHERE (:active IS NULL OR is_active = :active)
               AND (:class_name IS NULL OR class_name = :class_name COLLATE NOCASE)
               AND (:search IS NULL OR student_id LIKE :search ESCAPE '!' OR full_name LIKE :search ESCAPE '!'
                    OR email LIKE :search ESCAPE '!' OR model_class_label LIKE :search ESCAPE '!')
             ORDER BY class_name COLLATE NOCASE, section COLLATE NOCASE, full_name COLLATE NOCASE
            """,
            {"active": None if active is None else (1 if active else 0), "class_name": class_name,
             "search": _like(search)},
        )

    def class_names(self) -> list[str]:
        return [r["class_name"] for r in
                self.db.query_all("SELECT DISTINCT class_name FROM students ORDER BY class_name COLLATE NOCASE")]

    def count(self, *, active_only: bool = True) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM students" + (" WHERE is_active = 1" if active_only else ""),
                                  default=0))

    def mapped_labels(self) -> dict[str, dict[str, Any]]:
        rows = self.db.query_all(f"SELECT {_STUDENT_COLUMNS} FROM students WHERE model_class_label IS NOT NULL")
        return {row["model_class_label"]: row for row in rows}


# ======================================================================================
# Users (demo login)
# ======================================================================================

_USER_SELECT = """
    SELECT u.id, u.username, u.display_name, u.role, u.student_pk, u.password_hash, u.is_active,
           u.created_at, u.last_login_at, st.student_id, st.is_active AS student_active
      FROM users u
      LEFT JOIN students st ON st.id = u.student_pk
"""


class UserRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, *, username: str, display_name: str, role: str, password_hash: str,
               student_pk: int | None = None, conn: sqlite3.Connection | None = None) -> int:
        sql = """
            INSERT INTO users (username, display_name, role, student_pk, password_hash, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
        """
        params = (username, display_name, role, student_pk, password_hash, now_iso())
        try:
            if conn is not None:
                return int(conn.execute(sql, params).lastrowid)
            with self.db.transaction() as tx:
                return int(tx.execute(sql, params).lastrowid)
        except sqlite3.IntegrityError as exc:
            if "users.username" in str(exc):
                raise DuplicateUserError(f"The username '{username}' is already taken.") from exc
            if "users.student_pk" in str(exc):
                raise DuplicateUserError("This student already has a login account.") from exc
            raise DatabaseError(f"The account violates a database rule: {exc}") from exc

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        return self.db.query_one(_USER_SELECT + " WHERE u.username = ? COLLATE NOCASE", (username.strip(),))

    def get(self, user_id: int) -> dict[str, Any] | None:
        return self.db.query_one(_USER_SELECT + " WHERE u.id = ?", (user_id,))

    def get_for_student(self, student_pk: int) -> dict[str, Any] | None:
        return self.db.query_one(_USER_SELECT + " WHERE u.student_pk = ?", (student_pk,))

    def list(self) -> list[dict[str, Any]]:
        return self.db.query_all(_USER_SELECT + " ORDER BY u.role DESC, u.username COLLATE NOCASE")

    def count(self, role: str | None = None) -> int:
        if role is None:
            return int(self.db.scalar("SELECT COUNT(*) FROM users", default=0))
        return int(self.db.scalar("SELECT COUNT(*) FROM users WHERE role = ?", (role,), default=0))

    def set_password(self, user_id: int, password_hash: str) -> None:
        with self.db.transaction() as conn:
            if conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)).rowcount == 0:
                raise NotFoundError("Account not found.")

    def set_active(self, user_id: int, active: bool) -> None:
        with self.db.transaction() as conn:
            conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if active else 0, user_id))

    def touch_login(self, user_id: int) -> None:
        with self.db.transaction() as conn:
            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now_iso(), user_id))


# ======================================================================================
# Attendance sessions
# ======================================================================================

_SESSION_SELECT = f"""
    SELECT se.id, se.session_name, se.session_date, se.class_name, se.section, se.start_time, se.end_time,
           se.status, se.created_by, se.closed_at, se.closed_by, se.created_at,
           (SELECT COUNT(*) FROM attendance a WHERE a.session_id = se.id AND a.status IN ('present', 'late')) AS attended_count,
           (SELECT COUNT(*) FROM attendance a WHERE a.session_id = se.id AND a.status = 'present') AS present_count,
           (SELECT COUNT(*) FROM attendance a WHERE a.session_id = se.id AND a.status = 'late') AS late_count,
           (SELECT COUNT(*) FROM attendance a WHERE a.session_id = se.id AND a.status = 'absent') AS absent_count,
           (SELECT COUNT(*) FROM attendance a WHERE a.session_id = se.id AND a.status = 'excused') AS excused_count,
           (SELECT COUNT(*) FROM students st WHERE {ROSTER_CONDITIONS}
               AND NOT EXISTS (SELECT 1 FROM attendance a WHERE a.session_id = se.id AND a.student_id = st.id))
               AS not_marked_count,
           (SELECT COUNT(*) FROM students st WHERE {ROSTER_CONDITIONS}) AS roster_count
      FROM attendance_sessions se
"""


class SessionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, data: SessionInput, *, created_by: str) -> int:
        try:
            with self.db.transaction() as conn:
                return int(conn.execute(
                    """
                    INSERT INTO attendance_sessions (session_name, session_date, class_name, section, start_time,
                                                     end_time, status, created_by, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)
                    """,
                    (data.session_name, data.session_date, data.class_name, data.section, data.start_time,
                     data.end_time, created_by, now_iso()),
                ).lastrowid)
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc).upper():
                raise DuplicateSessionError(
                    f"A session named '{data.session_name}' already exists for {data.class_name} on {data.session_date}."
                ) from exc
            raise DatabaseError(f"The session violates a database rule: {exc}") from exc

    def get(self, session_id: int) -> dict[str, Any] | None:
        return self.db.query_one(_SESSION_SELECT + " WHERE se.id = :sid", {"sid": session_id})

    def list(self, *, date_from: str | None = None, date_to: str | None = None, class_name: str | None = None,
             status: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        return self.db.query_all(
            _SESSION_SELECT + """
             WHERE (:date_from IS NULL OR se.session_date >= :date_from)
               AND (:date_to IS NULL OR se.session_date <= :date_to)
               AND (:class_name IS NULL OR se.class_name = :class_name COLLATE NOCASE)
               AND (:status IS NULL OR se.status = :status)
             ORDER BY se.session_date DESC, se.start_time DESC, se.id DESC
             LIMIT :limit
            """,
            {"date_from": date_from, "date_to": date_to, "class_name": class_name, "status": status,
             "limit": int(limit)},
        )

    def roster(self, session_id: int) -> list[dict[str, Any]]:
        """Students expected at the session, with their attendance record if one exists."""
        return self.db.query_all(
            f"""
            SELECT st.id AS student_pk, st.student_id, st.full_name, st.class_name, st.section,
                   st.model_class_label, st.photo_path, st.consent_recorded_at,
                   a.id AS attendance_id, a.status, a.method, a.confidence, a.recorded_at, a.confirmed_by
              FROM attendance_sessions se
              JOIN students st ON {ROSTER_CONDITIONS}
              LEFT JOIN attendance a ON a.session_id = se.id AND a.student_id = st.id
             WHERE se.id = :sid
             ORDER BY st.full_name COLLATE NOCASE
            """,
            {"sid": session_id},
        )

    def close(self, session_id: int, *, closed_by: str) -> int:
        """Finalise: record 'absent' for each roster student without a record. Returns rows inserted."""
        now = now_iso()
        with self.db.transaction() as conn:
            row = conn.execute("SELECT status FROM attendance_sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                raise NotFoundError("Session not found.")
            if row["status"] == "closed":
                raise SessionClosedError("This session has already been finalised.")
            cursor = conn.execute(
                f"""
                INSERT INTO attendance (student_id, session_id, status, confidence, method, confirmed_by,
                                        recorded_at, notes)
                SELECT st.id, se.id, 'absent', NULL, 'session_finalize', :by, :now,
                       'No attendance recorded before the session was finalised'
                  FROM attendance_sessions se
                  JOIN students st ON {ROSTER_CONDITIONS}
                 WHERE se.id = :sid
                   AND NOT EXISTS (SELECT 1 FROM attendance a WHERE a.session_id = se.id AND a.student_id = st.id)
                """,
                {"sid": session_id, "by": closed_by, "now": now},
            )
            inserted = cursor.rowcount if cursor.rowcount is not None else 0
            conn.execute("UPDATE attendance_sessions SET status = 'closed', closed_at = ?, closed_by = ? WHERE id = ?",
                         (now, closed_by, session_id))
            return int(inserted)


# ======================================================================================
# Attendance
# ======================================================================================


@dataclass(frozen=True)
class RecordOutcome:
    inserted: bool
    attendance_id: int | None
    existing_status: str | None = None


_RECORD_SELECT = """
    SELECT a.id AS attendance_id, a.session_id, a.student_id AS student_pk,
           se.session_date, se.session_name, se.class_name AS session_class, se.section AS session_section,
           se.start_time, se.status AS session_status,
           st.student_id, st.full_name, st.class_name, st.section, st.photo_path,
           a.status, a.confidence, a.method, a.confirmed_by, a.recorded_at, a.updated_at, a.notes
      FROM attendance a
      JOIN students st ON st.id = a.student_id
      JOIN attendance_sessions se ON se.id = a.session_id
"""


class AttendanceRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, *, student_pk: int, session_id: int, status: str, method: str, confirmed_by: str,
               confidence: float | None = None, notes: str | None = None) -> RecordOutcome:
        """Insert one row. Duplicates are refused by SQLite's UNIQUE(student_id, session_id)."""
        if status not in ATTENDANCE_STATUSES:
            raise ValueError(f"Invalid status {status!r}.")
        if method not in ATTENDANCE_METHODS:
            raise ValueError(f"Invalid method {method!r}.")
        with self.db.transaction() as conn:
            session = conn.execute("SELECT status FROM attendance_sessions WHERE id = ?", (session_id,)).fetchone()
            if session is None:
                raise NotFoundError("Attendance session not found.")
            if session["status"] != "open":
                raise SessionClosedError("This session has been finalised. Use a correction on Attendance Records.")
            student = conn.execute("SELECT is_active FROM students WHERE id = ?", (student_pk,)).fetchone()
            if student is None:
                raise NotFoundError("Student not found.")
            if int(student["is_active"]) != 1:
                raise InactiveStudentError("This student is deactivated.")
            cursor = conn.execute(
                """
                INSERT INTO attendance (student_id, session_id, status, confidence, method, confirmed_by,
                                        recorded_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (student_id, session_id) DO NOTHING
                """,
                (student_pk, session_id, status, confidence, method, confirmed_by, now_iso(), notes),
            )
            if cursor.rowcount == 1:
                return RecordOutcome(True, int(cursor.lastrowid))
            existing = conn.execute("SELECT id, status FROM attendance WHERE student_id = ? AND session_id = ?",
                                    (student_pk, session_id)).fetchone()
            return RecordOutcome(False, int(existing["id"]) if existing else None,
                                 existing["status"] if existing else None)

    def delete_face_record(self, attendance_id: int) -> dict[str, Any]:
        """Undo an automatic / face-recognition mark in an OPEN session. Returns the removed row."""
        with self.db.transaction() as conn:
            row = conn.execute(
                """
                SELECT a.id, a.student_id, a.session_id, a.method, a.confidence, se.status AS session_status,
                       st.model_class_label
                  FROM attendance a
                  JOIN attendance_sessions se ON se.id = a.session_id
                  JOIN students st ON st.id = a.student_id
                 WHERE a.id = ?
                """,
                (attendance_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError("The attendance record no longer exists.")
            if row["method"] != "face_recognition":
                raise DatabaseError("Only face-recognition marks can be undone. Use a correction instead.")
            if row["session_status"] != "open":
                raise SessionClosedError("The session is finalised. Use a correction on Attendance Records.")
            conn.execute("DELETE FROM attendance WHERE id = ?", (attendance_id,))
            return dict(row)

    def get_for(self, student_pk: int, session_id: int) -> dict[str, Any] | None:
        return self.db.query_one(_RECORD_SELECT + " WHERE a.student_id = ? AND a.session_id = ?", (student_pk, session_id))

    def correct_status(self, attendance_id: int, *, new_status: str, reason: str, changed_by: str) -> bool:
        if new_status not in ATTENDANCE_STATUSES:
            raise ValueError(f"Invalid status {new_status!r}.")
        now = now_iso()
        with self.db.transaction() as conn:
            row = conn.execute("SELECT status FROM attendance WHERE id = ?", (attendance_id,)).fetchone()
            if row is None:
                raise NotFoundError("Attendance record not found.")
            if row["status"] == new_status:
                return False
            conn.execute("UPDATE attendance SET status = ?, updated_at = ? WHERE id = ?", (new_status, now, attendance_id))
            conn.execute(
                "INSERT INTO attendance_corrections (attendance_id, previous_status, new_status, reason, changed_by, "
                "changed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (attendance_id, row["status"], new_status, reason.strip(), changed_by, now),
            )
            return True

    def search(self, *, date_from: str | None = None, date_to: str | None = None, class_name: str | None = None,
               statuses: Iterable[str] | None = None, methods: Iterable[str] | None = None,
               student_pk: int | None = None, session_id: int | None = None, search: str | None = None,
               limit: int = 5000) -> list[dict[str, Any]]:
        status_list, method_list = list(statuses or []), list(methods or [])
        status_sql, status_params = _in_clause("st_", status_list)
        method_sql, method_params = _in_clause("me_", method_list)
        params: dict[str, Any] = {
            "date_from": date_from, "date_to": date_to, "class_name": class_name, "student_pk": student_pk,
            "session_id": session_id, "search": _like(search), "use_status": 1 if status_list else 0,
            "use_method": 1 if method_list else 0, "limit": int(limit), **status_params, **method_params,
        }
        return self.db.query_all(
            _RECORD_SELECT + f"""
             WHERE (:date_from IS NULL OR se.session_date >= :date_from)
               AND (:date_to IS NULL OR se.session_date <= :date_to)
               AND (:class_name IS NULL OR se.class_name = :class_name COLLATE NOCASE)
               AND (:student_pk IS NULL OR a.student_id = :student_pk)
               AND (:session_id IS NULL OR a.session_id = :session_id)
               AND (:use_status = 0 OR a.status IN {status_sql})
               AND (:use_method = 0 OR a.method IN {method_sql})
               AND (:search IS NULL OR st.student_id LIKE :search ESCAPE '!' OR st.full_name LIKE :search ESCAPE '!'
                    OR se.session_name LIKE :search ESCAPE '!' OR a.confirmed_by LIKE :search ESCAPE '!')
             ORDER BY se.session_date DESC, a.recorded_at DESC, a.id DESC
             LIMIT :limit
            """,
            params,
        )

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.db.query_all(_RECORD_SELECT + " ORDER BY a.recorded_at DESC, a.id DESC LIMIT ?", (int(limit),))

    def corrections(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.db.query_all(
            """
            SELECT c.id, c.attendance_id, c.previous_status, c.new_status, c.reason, c.changed_by, c.changed_at,
                   st.student_id, st.full_name, se.session_name, se.session_date
              FROM attendance_corrections c
              JOIN attendance a ON a.id = c.attendance_id
              JOIN students st ON st.id = a.student_id
              JOIN attendance_sessions se ON se.id = a.session_id
             ORDER BY c.changed_at DESC, c.id DESC
             LIMIT ?
            """,
            (int(limit),),
        )

    def report_rows(self, *, date_from: str, date_to: str, class_name: str | None = None,
                    finalized_only: bool = True, student_pk: int | None = None) -> list[dict[str, Any]]:
        return self.db.query_all(
            """
            SELECT se.id AS session_id, se.session_date, se.session_name, se.class_name AS session_class,
                   se.section AS session_section, se.status AS session_status, se.start_time,
                   st.id AS student_pk, st.student_id, st.full_name, st.class_name, st.section,
                   a.status, a.method, a.recorded_at, a.confidence
              FROM attendance a
              JOIN attendance_sessions se ON se.id = a.session_id
              JOIN students st ON st.id = a.student_id
             WHERE se.session_date BETWEEN :date_from AND :date_to
               AND (:class_name IS NULL OR se.class_name = :class_name COLLATE NOCASE)
               AND (:finalized_only = 0 OR se.status = 'closed')
               AND (:student_pk IS NULL OR a.student_id = :student_pk)
            """,
            {"date_from": date_from, "date_to": date_to, "class_name": class_name,
             "finalized_only": 1 if finalized_only else 0, "student_pk": student_pk},
        )


# ======================================================================================
# Recognition logs and settings
# ======================================================================================


class RecognitionLogRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, *, recognition_status: str, predicted_label: str | None, confidence: float | None,
            student_pk: int | None = None, session_id: int | None = None, source: str | None = None) -> None:
        if confidence is not None:
            confidence = min(max(float(confidence), 0.0), 1.0)
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO recognition_logs (student_id, session_id, predicted_label, confidence, "
                "recognition_status, source, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (student_pk, session_id, predicted_label, confidence, recognition_status, source, now_iso()),
            )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.db.query_all(
            """
            SELECT l.timestamp, l.recognition_status, l.predicted_label, l.confidence, l.source,
                   st.student_id, st.full_name, se.session_name
              FROM recognition_logs l
              LEFT JOIN students st ON st.id = l.student_id
              LEFT JOIN attendance_sessions se ON se.id = l.session_id
             ORDER BY l.timestamp DESC, l.id DESC
             LIMIT ?
            """,
            (int(limit),),
        )

    def delete_older_than(self, cutoff_iso: str) -> int:
        with self.db.transaction() as conn:
            return int(conn.execute("DELETE FROM recognition_logs WHERE timestamp < ?", (cutoff_iso,)).rowcount or 0)


class SettingsRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_all(self) -> dict[str, str]:
        return {r["setting_key"]: r["setting_value"]
                for r in self.db.query_all("SELECT setting_key, setting_value FROM app_settings")}

    def set_many(self, values: dict[str, str]) -> None:
        now = now_iso()
        with self.db.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO app_settings (setting_key, setting_value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT (setting_key) DO UPDATE SET setting_value = excluded.setting_value,
                                                        updated_at = excluded.updated_at
                """,
                [(k, v, now) for k, v in values.items()],
            )


def table_counts(db: Database) -> dict[str, int]:
    names = ("students", "users", "attendance_sessions", "attendance", "recognition_logs")
    with db.connect() as conn:
        # Table names come from the constant tuple above, never from user input.
        return {name: int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]) for name in names}
