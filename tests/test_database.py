import sqlite3
from datetime import date

import pytest

from database.db import Database
from database.repositories import (
    AttendanceRepository,
    DuplicateStudentError,
    DuplicateUserError,
    ModelLabelConflictError,
    SessionClosedError,
    SessionRepository,
    StudentRepository,
    UserRepository,
)
from database.schema import LATEST_SCHEMA_VERSION, SchemaVersionError, current_schema_version, initialize_database
from services.attendance_service import AttendanceService
from services.auth_service import AuthService
from utils.validators import validate_session, validate_student


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "attendance.db"
    initialize_database(path)
    return Database(path)


def add_student(db, sid, cls="TECH 405", label=None, consent=True):
    data = validate_student(student_id=sid, full_name=sid, class_name=cls, model_class_label=label,
                            known_model_labels={label} if label else None)
    return StudentRepository(db).create(data, consent=consent)


def add_session(db, cls="TECH 405", name="Lecture"):
    return SessionRepository(db).create(validate_session(session_name=name, session_date=date.today(), class_name=cls),
                                        created_by="teacher")


def test_initialisation_is_idempotent(tmp_path):
    path = tmp_path / "a.db"
    assert initialize_database(path).created
    assert not initialize_database(path).applied
    assert current_schema_version(path) == LATEST_SCHEMA_VERSION


def test_foreign_database_is_left_untouched(tmp_path):
    path = tmp_path / "other.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE x (y INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError):
        initialize_database(path)


def test_unique_student_id_and_model_label(db):
    add_student(db, "Amit", label="Amit")
    with pytest.raises(DuplicateStudentError):
        add_student(db, "amit")
    with pytest.raises(ModelLabelConflictError):
        add_student(db, "Other", label="Amit")


def test_duplicate_attendance_blocked_by_sqlite(db):
    pk, sid = add_student(db, "Amit"), add_session(db)
    repo = AttendanceRepository(db)
    first = repo.record(student_pk=pk, session_id=sid, status="present", method="face_recognition",
                        confirmed_by="teacher", confidence=0.97)
    second = repo.record(student_pk=pk, session_id=sid, status="late", method="manual", confirmed_by="teacher")
    assert first.inserted and not second.inserted and second.existing_status == "present"
    with db.connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO attendance (student_id, session_id, status, method, confirmed_by, recorded_at) "
                     "VALUES (?, ?, 'present', 'manual', 'x', '2026-01-01T00:00:00')", (pk, sid))


def test_face_recognition_requires_valid_confidence(db):
    pk, sid = add_student(db, "Amit"), add_session(db)
    with pytest.raises(sqlite3.IntegrityError):
        AttendanceRepository(db).record(student_pk=pk, session_id=sid, status="present", method="face_recognition",
                                        confirmed_by="t", confidence=None)


def test_auto_mark_flow_and_undo(db):
    add_student(db, "Amit", label="Amit")
    sid = add_session(db)
    service = AttendanceService(db)
    first = service.mark_recognised(label="Amit", confidence=0.98, session_id=sid, status="present",
                                    operator="teacher", source="live_video", automatic=True)
    again = service.mark_recognised(label="Amit", confidence=0.99, session_id=sid, status="present",
                                    operator="teacher", source="live_video", automatic=True)
    assert first.ok and not again.ok and again.duplicate
    assert service.undo_mark(first.attendance_id).ok
    assert service.mark_recognised(label="Amit", confidence=0.95, session_id=sid, status="present",
                                   operator="teacher", source="live_video", automatic=True).ok


def test_unconsented_or_unlinked_faces_are_not_marked(db):
    add_student(db, "Amit", label="Amit", consent=False)
    sid = add_session(db)
    service = AttendanceService(db)
    assert not service.mark_recognised(label="Amit", confidence=0.99, session_id=sid, status="present",
                                       operator="t", source="live_video", automatic=True).ok
    assert not service.mark_recognised(label="Nobody", confidence=0.99, session_id=sid, status="present",
                                       operator="t", source="live_video", automatic=True).ok


def test_ending_session_marks_only_unrecorded_roster_absent(db):
    present = add_student(db, "Amit")
    add_student(db, "Udhav")
    add_student(db, "Guest", cls="OTHER")
    sid = add_session(db)
    AttendanceRepository(db).record(student_pk=present, session_id=sid, status="present", method="manual",
                                    confirmed_by="t")
    assert SessionRepository(db).close(sid, closed_by="t") == 1
    session = SessionRepository(db).get(sid)
    assert session["status"] == "closed" and session["absent_count"] == 1 and session["attended_count"] == 1
    with pytest.raises(SessionClosedError):
        AttendanceRepository(db).record(student_pk=present, session_id=sid, status="late", method="manual",
                                        confirmed_by="t")


def test_login_roles_and_passwords(db):
    auth = AuthService(db)
    auth.create_teacher("teacher", "Teacher", "teacher123", "teacher123")
    pk = add_student(db, "Amit")
    auth.create_student_login(student_pk=pk, username="Amit", display_name="Amit", password="student123")
    assert auth.authenticate("teacher", "teacher123", "teacher").user.is_teacher
    assert auth.authenticate("amit", "student123", "student").user.student_pk == pk
    assert auth.authenticate("Amit", "wrong", "student").user is None
    assert auth.authenticate("Amit", "student123", "teacher").user is None  # wrong tab
    with pytest.raises(DuplicateUserError):
        UserRepository(db).create(username="AMIT", display_name="x", role="teacher", password_hash="x")
    StudentRepository(db).set_active(pk, False)
    assert auth.authenticate("Amit", "student123", "student").user is None  # deactivated student
