"""Demo login for teachers and students (academic demonstration only).

* Passwords are stored as salted PBKDF2 hashes in the SQLite ``users`` table.
* Teacher: full access. Student: sees only their own attendance.
* Demo credentials are defined in utils/config.py and shown on the sign-in page; change them on
  the "My account" page. This is not production-grade authentication.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from database.db import Database
from database.repositories import UserRepository
from utils.config import DEMO_STUDENT_PASSWORD
from utils.security import dummy_verify, hash_password, verify_password
from utils.validators import ValidationError, validate_new_password, validate_username


@dataclass(frozen=True)
class CurrentUser:
    id: int
    username: str
    display_name: str
    role: str  # "teacher" or "student"
    student_pk: int | None

    @property
    def is_teacher(self) -> bool:
        return self.role == "teacher"


@dataclass(frozen=True)
class LoginResult:
    user: CurrentUser | None
    message: str


class AuthService:
    def __init__(self, db: Database) -> None:
        self.users = UserRepository(db)

    def authenticate(self, username: str, password: str, role: str) -> LoginResult:
        row = self.users.get_by_username(username or "")
        if row is None:
            dummy_verify()
            return LoginResult(None, "Invalid username or password.")
        if not verify_password(password, row["password_hash"]):
            return LoginResult(None, "Invalid username or password.")
        if row["role"] != role:
            other = "Teacher" if row["role"] == "teacher" else "Student"
            return LoginResult(None, f"This is a {other.lower()} account — use the {other} tab.")
        if not row["is_active"] or (role == "student" and not row["student_active"]):
            return LoginResult(None, "This account is disabled. Please contact the teacher.")
        self.users.touch_login(int(row["id"]))
        user = CurrentUser(int(row["id"]), row["username"], row["display_name"], row["role"], row["student_pk"])
        return LoginResult(user, f"Welcome, {user.display_name}!")

    def change_password(self, user_id: int, current: str, new: str, confirm: str) -> None:
        row = self.users.get(user_id)
        if row is None or not verify_password(current, row["password_hash"]):
            raise ValidationError(["Current password is incorrect."])
        validate_new_password(new, confirm)
        self.users.set_password(user_id, hash_password(new))

    def reset_password(self, user_id: int, new_password: str = DEMO_STUDENT_PASSWORD) -> None:
        """Teacher action: reset an account to a known password (default: the demo student password)."""
        validate_new_password(new_password, new_password)
        self.users.set_password(user_id, hash_password(new_password))

    def create_teacher(self, username: str, display_name: str, password: str, confirm: str) -> int:
        name = validate_username(username)
        validate_new_password(password, confirm)
        display = " ".join((display_name or "").split()) or name
        return self.users.create(username=name, display_name=display, role="teacher",
                                 password_hash=hash_password(password))

    def create_student_login(self, *, student_pk: int, username: str, display_name: str,
                             password: str = DEMO_STUDENT_PASSWORD, conn: sqlite3.Connection | None = None) -> int:
        return self.users.create(username=validate_username(username), display_name=display_name, role="student",
                                 password_hash=hash_password(password), student_pk=student_pk, conn=conn)

    def list_accounts(self) -> list[dict]:
        return [{k: v for k, v in row.items() if k != "password_hash"} for row in self.users.list()]

    def account_for_student(self, student_pk: int) -> dict | None:
        row = self.users.get_for_student(student_pk)
        return None if row is None else {k: v for k, v in row.items() if k != "password_hash"}

    def set_active(self, user_id: int, active: bool) -> None:
        self.users.set_active(user_id, active)
