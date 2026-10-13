"""Input validation for students, sessions and accounts (friendly messages for the UI).

The SQLite schema enforces the same rules with CHECK / UNIQUE constraints as the final authority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time

STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,31}$")
USERNAME_PATTERN = STUDENT_ID_PATTERN
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

MAX_NAME_LENGTH = 120
MAX_CLASS_LENGTH = 50
MAX_SECTION_LENGTH = 20
MAX_SESSION_NAME_LENGTH = 80
MIN_PASSWORD_LENGTH = 6


class ValidationError(ValueError):
    """Invalid user input. ``errors`` holds one message per problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


@dataclass(frozen=True)
class StudentInput:
    student_id: str
    full_name: str
    class_name: str
    section: str | None
    email: str | None
    model_class_label: str | None


def validate_student(
    *,
    student_id: str,
    full_name: str,
    class_name: str,
    section: str | None = None,
    email: str | None = None,
    model_class_label: str | None = None,
    known_model_labels: set[str] | None = None,
) -> StudentInput:
    """Validate and normalise student fields. Raises :class:`ValidationError`."""
    errors: list[str] = []
    sid = (student_id or "").strip()
    if not sid:
        errors.append("Student ID is required.")
    elif not STUDENT_ID_PATTERN.match(sid):
        errors.append("Student ID must be 1–32 characters: letters, digits, '.', '_' or '-'.")

    name = _clean(full_name)
    if not name:
        errors.append("Full name is required.")
    elif len(name) > MAX_NAME_LENGTH:
        errors.append(f"Full name must be at most {MAX_NAME_LENGTH} characters.")

    cls = _clean(class_name)
    if not cls:
        errors.append("Class is required.")
    elif len(cls) > MAX_CLASS_LENGTH:
        errors.append(f"Class must be at most {MAX_CLASS_LENGTH} characters.")

    sec = _clean(section) or None
    if sec and len(sec) > MAX_SECTION_LENGTH:
        errors.append(f"Section must be at most {MAX_SECTION_LENGTH} characters.")

    mail = (email or "").strip() or None
    if mail and not EMAIL_PATTERN.match(mail):
        errors.append("Email address is not valid.")

    label = (model_class_label or "").strip() or None
    if label is not None:
        if known_model_labels is None:
            errors.append("models/class_names.json was not found, so a model class cannot be linked yet.")
        elif label not in known_model_labels:
            errors.append(f"Model class '{label}' does not exist in the trained model.")

    if errors:
        raise ValidationError(errors)
    return StudentInput(sid, name, cls, sec, mail.lower() if mail else None, label)


@dataclass(frozen=True)
class SessionInput:
    session_name: str
    session_date: str
    class_name: str
    section: str | None
    start_time: str | None
    end_time: str | None


def _format_time(value: time | str | None) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, time):
        return value.strftime("%H:%M")
    return str(value).strip()


def validate_session(
    *,
    session_name: str,
    session_date: date | str,
    class_name: str,
    section: str | None = None,
    start_time: time | str | None = None,
    end_time: time | str | None = None,
) -> SessionInput:
    errors: list[str] = []
    name = _clean(session_name)
    if not name:
        errors.append("Session name is required.")
    elif len(name) > MAX_SESSION_NAME_LENGTH:
        errors.append(f"Session name must be at most {MAX_SESSION_NAME_LENGTH} characters.")

    if isinstance(session_date, date):
        iso_date = session_date.isoformat()
    else:
        iso_date = str(session_date).strip()
        try:
            iso_date = date.fromisoformat(iso_date).isoformat()
        except ValueError:
            errors.append("Session date must be in YYYY-MM-DD format.")

    cls = _clean(class_name)
    if not cls:
        errors.append("Class is required.")
    elif len(cls) > MAX_CLASS_LENGTH:
        errors.append(f"Class must be at most {MAX_CLASS_LENGTH} characters.")

    sec = _clean(section) or None
    if sec and len(sec) > MAX_SECTION_LENGTH:
        errors.append(f"Section must be at most {MAX_SECTION_LENGTH} characters.")

    start, end = _format_time(start_time), _format_time(end_time)
    for label, value in (("Start time", start), ("End time", end)):
        if value is not None and not TIME_PATTERN.match(value):
            errors.append(f"{label} must be in HH:MM (24-hour) format.")
    if start and end and TIME_PATTERN.match(start) and TIME_PATTERN.match(end) and end <= start:
        errors.append("End time must be after start time.")

    if errors:
        raise ValidationError(errors)
    return SessionInput(name, iso_date, cls, sec, start, end)


def validate_username(username: str) -> str:
    name = (username or "").strip()
    if not USERNAME_PATTERN.match(name):
        raise ValidationError(["Username must be 1–32 characters: letters, digits, '.', '_' or '-'."])
    return name


def validate_new_password(password: str, confirm: str) -> str:
    errors: list[str] = []
    if len(password or "") < MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if password != confirm:
        errors.append("Passwords do not match.")
    if errors:
        raise ValidationError(errors)
    return password


def now_iso() -> str:
    """Current local time, ISO-8601 with seconds (stored in SQLite)."""
    return datetime.now().isoformat(timespec="seconds")


def today_iso() -> str:
    return date.today().isoformat()
