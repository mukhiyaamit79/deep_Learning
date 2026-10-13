"""First-start demo data: the teacher account and one student per trained model class.

Runs on every app start but only ever ADDS missing things:
* a teacher account if no teacher exists (demo credentials from utils/config.py);
* if the students table is EMPTY: one student per class in models/class_names.json (falling back
  to the dataset/ folder names), linked to that class, with the first training photo as profile
  picture and a student login (username = Student ID, demo password).

Seeded students are the project members whose consented photos are in dataset/, so their
face-recognition consent is recorded at seeding. The teacher can withdraw it on the Students page.
Existing data is never changed or deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from database.db import Database, DatabaseError
from database.repositories import StudentRepository, UserRepository
from services.auth_service import AuthService
from services.face_recognition import ModelArtifactError, read_class_names
from services.student_service import first_dataset_photo, to_stored_path
from utils.config import (
    DEFAULT_CLASS_NAME,
    DEMO_TEACHER_NAME,
    DEMO_TEACHER_PASSWORD,
    DEMO_TEACHER_USERNAME,
    AppPaths,
)
from utils.security import hash_password
from utils.validators import ValidationError, validate_student


@dataclass
class SeedResult:
    teacher_created: bool = False
    students_created: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def _labels(paths: AppPaths) -> tuple[list[str], bool]:
    """Class labels and whether they come from the trained model (True) or only from dataset folders."""
    try:
        return list(read_class_names(paths.class_names_path)), True
    except ModelArtifactError:
        pass
    if paths.dataset_dir.is_dir():
        return sorted(p.name for p in paths.dataset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")), False
    return [], False


def seed_demo_data(db: Database, paths: AppPaths) -> SeedResult:
    result = SeedResult()
    users = UserRepository(db)
    students = StudentRepository(db)
    auth = AuthService(db)

    if users.count("teacher") == 0:
        users.create(username=DEMO_TEACHER_USERNAME, display_name=DEMO_TEACHER_NAME, role="teacher",
                     password_hash=hash_password(DEMO_TEACHER_PASSWORD))
        result.teacher_created = True

    if students.count(active_only=False) > 0:
        return result

    labels, from_model = _labels(paths)
    for label in labels:
        try:
            data = validate_student(student_id=label, full_name=label, class_name=DEFAULT_CLASS_NAME,
                                    model_class_label=label if from_model else None,
                                    known_model_labels=set(labels) if from_model else None)
        except ValidationError as exc:
            result.messages.append(f"Skipped '{label}': {exc}")
            continue
        photo = to_stored_path(first_dataset_photo(paths.dataset_dir, label))
        try:
            with db.transaction() as tx:
                pk = students.create(data, consent=True, photo_path=photo, conn=tx)
                auth.create_student_login(student_pk=pk, username=data.student_id, display_name=data.full_name,
                                          conn=tx)
        except DatabaseError as exc:  # e.g. a student ID equal to an existing username
            result.messages.append(f"Skipped '{label}': {exc}")
            continue
        result.students_created.append(label)
    if labels and not from_model:
        result.messages.append("models/class_names.json not found: students were created from dataset/ folders "
                               "and are not linked to the model yet.")
    return result
