"""Database schema, versioned migrations and safe initialisation.

* ``PRAGMA user_version`` stores the schema version; ``MIGRATIONS`` is append-only.
* Each migration runs once in its own transaction and is recorded in ``schema_migrations``.
* Before upgrading an EXISTING database, a timestamped backup is written to ``backups/``.
* Startup never deletes or recreates a database. A database from a newer app version, or a
  non-empty SQLite file that is not a Smart Attendance database, is refused without modification.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from database.db import Database, DatabaseError
from utils.validators import now_iso

_LOCAL_NOW = "(strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime'))"


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        "initial schema: students, users, sessions, attendance, logs, settings",
        (
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)",
            f"""
            CREATE TABLE students (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id           TEXT    NOT NULL COLLATE NOCASE UNIQUE
                                     CHECK (length(trim(student_id)) BETWEEN 1 AND 32),
                full_name            TEXT    NOT NULL CHECK (length(trim(full_name)) BETWEEN 1 AND 120),
                class_name           TEXT    NOT NULL CHECK (length(trim(class_name)) BETWEEN 1 AND 50),
                section              TEXT    CHECK (section IS NULL OR length(section) <= 20),
                email                TEXT    CHECK (email IS NULL OR email LIKE '%_@_%._%'),
                model_class_label    TEXT    CHECK (model_class_label IS NULL OR length(model_class_label) >= 1),
                photo_path           TEXT,
                is_active            INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                consent_recorded_at  TEXT,
                created_at           TEXT    NOT NULL DEFAULT {_LOCAL_NOW},
                updated_at           TEXT    NOT NULL DEFAULT {_LOCAL_NOW}
            )
            """,
            # One trained model class belongs to at most one student record.
            "CREATE UNIQUE INDEX ux_students_model_class_label ON students (model_class_label) "
            "WHERE model_class_label IS NOT NULL",
            "CREATE INDEX ix_students_class ON students (class_name, section)",
            "CREATE INDEX ix_students_active ON students (is_active)",
            f"""
            CREATE TABLE users (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                username       TEXT    NOT NULL COLLATE NOCASE UNIQUE
                               CHECK (length(trim(username)) BETWEEN 1 AND 32),
                display_name   TEXT    NOT NULL CHECK (length(trim(display_name)) >= 1),
                role           TEXT    NOT NULL CHECK (role IN ('teacher', 'student')),
                student_pk     INTEGER UNIQUE REFERENCES students (id) ON DELETE CASCADE,
                password_hash  TEXT    NOT NULL,
                is_active      INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                created_at     TEXT    NOT NULL DEFAULT {_LOCAL_NOW},
                last_login_at  TEXT,
                CHECK ((role = 'student' AND student_pk IS NOT NULL) OR (role = 'teacher' AND student_pk IS NULL))
            )
            """,
            f"""
            CREATE TABLE attendance_sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name  TEXT    NOT NULL CHECK (length(trim(session_name)) BETWEEN 1 AND 80),
                session_date  TEXT    NOT NULL
                              CHECK (session_date GLOB '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]'),
                class_name    TEXT    NOT NULL CHECK (length(trim(class_name)) BETWEEN 1 AND 50),
                section       TEXT    CHECK (section IS NULL OR length(section) <= 20),
                start_time    TEXT    CHECK (start_time IS NULL OR start_time GLOB '[0-2][0-9]:[0-5][0-9]'),
                end_time      TEXT    CHECK (end_time IS NULL OR end_time GLOB '[0-2][0-9]:[0-5][0-9]'),
                status        TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
                created_by    TEXT,
                closed_at     TEXT,
                closed_by     TEXT,
                created_at    TEXT    NOT NULL DEFAULT {_LOCAL_NOW},
                CHECK (start_time IS NULL OR end_time IS NULL OR end_time > start_time)
            )
            """,
            "CREATE UNIQUE INDEX ux_sessions_identity ON attendance_sessions "
            "(session_date, class_name COLLATE NOCASE, COALESCE(section, ''), session_name COLLATE NOCASE)",
            "CREATE INDEX ix_sessions_date ON attendance_sessions (session_date)",
            "CREATE INDEX ix_sessions_status ON attendance_sessions (status)",
            f"""
            CREATE TABLE attendance (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id    INTEGER NOT NULL REFERENCES students (id) ON DELETE RESTRICT,
                session_id    INTEGER NOT NULL REFERENCES attendance_sessions (id) ON DELETE RESTRICT,
                status        TEXT    NOT NULL CHECK (status IN ('present', 'late', 'absent', 'excused')),
                confidence    REAL    CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
                method        TEXT    NOT NULL CHECK (method IN ('face_recognition', 'manual', 'session_finalize')),
                confirmed_by  TEXT    NOT NULL CHECK (length(trim(confirmed_by)) >= 1),
                recorded_at   TEXT    NOT NULL DEFAULT {_LOCAL_NOW},
                updated_at    TEXT,
                notes         TEXT,
                UNIQUE (student_id, session_id),
                CHECK (method <> 'face_recognition' OR confidence IS NOT NULL)
            )
            """,
            "CREATE INDEX ix_attendance_session ON attendance (session_id)",
            "CREATE INDEX ix_attendance_status ON attendance (status)",
            "CREATE INDEX ix_attendance_recorded_at ON attendance (recorded_at)",
            f"""
            CREATE TABLE attendance_corrections (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                attendance_id    INTEGER NOT NULL REFERENCES attendance (id) ON DELETE CASCADE,
                previous_status  TEXT    NOT NULL,
                new_status       TEXT    NOT NULL,
                reason           TEXT    NOT NULL CHECK (length(trim(reason)) >= 3),
                changed_by       TEXT    NOT NULL,
                changed_at       TEXT    NOT NULL DEFAULT {_LOCAL_NOW}
            )
            """,
            "CREATE INDEX ix_corrections_attendance ON attendance_corrections (attendance_id)",
            f"""
            CREATE TABLE recognition_logs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id          INTEGER REFERENCES students (id) ON DELETE SET NULL,
                session_id          INTEGER REFERENCES attendance_sessions (id) ON DELETE SET NULL,
                predicted_label     TEXT,
                confidence          REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
                recognition_status  TEXT NOT NULL CHECK (recognition_status IN (
                                        'confirmed', 'auto_marked', 'duplicate', 'undone',
                                        'rejected_by_operator', 'unknown', 'uncertain', 'unmapped',
                                        'not_consented', 'inactive', 'session_closed', 'error')),
                source              TEXT CHECK (source IS NULL OR source IN ('live_video', 'still_image')),
                timestamp           TEXT NOT NULL DEFAULT {_LOCAL_NOW}
            )
            """,
            "CREATE INDEX ix_recognition_logs_timestamp ON recognition_logs (timestamp)",
            f"""
            CREATE TABLE app_settings (
                setting_key    TEXT PRIMARY KEY,
                setting_value  TEXT NOT NULL,
                updated_at     TEXT NOT NULL DEFAULT {_LOCAL_NOW}
            )
            """,
        ),
    ),
)

LATEST_SCHEMA_VERSION: int = MIGRATIONS[-1].version


class SchemaVersionError(DatabaseError):
    """The database file cannot be used safely by this version of the application."""


@dataclass(frozen=True)
class InitResult:
    created: bool
    from_version: int
    to_version: int
    applied: tuple[str, ...]
    backup_path: Path | None


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchall()
    return [row[0] for row in rows]


def backup_database(db_path: Path, backups_dir: Path, label: str) -> Path:
    backups_dir.mkdir(parents=True, exist_ok=True)
    target = backups_dir / f"{db_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}.db"
    src, dst = sqlite3.connect(str(db_path)), sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def initialize_database(db_path: Path, backups_dir: Path | None = None) -> InitResult:
    """Create the database on first run and apply pending migrations. Safe on every start."""
    db_path = Path(db_path)
    backups_dir = backups_dir or db_path.parent / "backups"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    existed = db_path.exists() and db_path.stat().st_size > 0
    database = Database(db_path)

    with database.connect() as conn:
        current = _user_version(conn)
        if current > LATEST_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"The database at {db_path} uses schema version {current}, newer than this app supports "
                f"({LATEST_SCHEMA_VERSION}). The database was NOT modified."
            )
        if current == 0 and existed and _user_tables(conn):
            raise SchemaVersionError(
                f"{db_path} is an existing SQLite database that was not created by this app "
                f"(tables: {', '.join(_user_tables(conn))}). Choose another path. The file was NOT modified."
            )

    pending = [m for m in MIGRATIONS if m.version > current]
    backup_path: Path | None = None
    if pending and existed and current > 0:
        backup_path = backup_database(db_path, backups_dir, f"before_v{pending[0].version}")

    applied: list[str] = []
    for migration in pending:
        with database.transaction() as conn:
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (migration.version, migration.name, now_iso()),
            )
            conn.execute(f"PRAGMA user_version = {int(migration.version)}")  # trusted int constant
        applied.append(f"v{migration.version}: {migration.name}")

    if not existed:
        with database.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")

    return InitResult(not existed, current, LATEST_SCHEMA_VERSION, tuple(applied), backup_path)


def current_schema_version(db_path: Path) -> int:
    with Database(db_path).connect() as conn:
        return _user_version(conn)
