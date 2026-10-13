"""Student management, student photos, login accounts and the student <-> model-class link.

Registering a student does NOT teach the CNN a new face: the model only knows the classes in
models/class_names.json (Amit, Shreejal, Susman, Udhav, Uttam). A student is recognisable only
when the record is (1) linked to a trained class, (2) active and (3) has consent recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from database.db import Database
from database.repositories import StudentRepository
from services.auth_service import AuthService
from utils.config import PROJECT_ROOT
from utils.validators import StudentInput, validate_student

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class RecognitionEligibility(str, Enum):
    RECOGNIZABLE = "recognizable"
    NOT_IN_MODEL = "not_in_model"
    MODEL_UNAVAILABLE = "model_unavailable"
    STALE_MAPPING = "stale_mapping"
    NO_CONSENT = "no_consent"
    INACTIVE = "inactive"


ELIGIBILITY_TEXT: dict[RecognitionEligibility, str] = {
    RecognitionEligibility.RECOGNIZABLE: "Recognisable by the camera",
    RecognitionEligibility.NOT_IN_MODEL: "Registered – not in trained model",
    RecognitionEligibility.MODEL_UNAVAILABLE: "Linked – model files missing",
    RecognitionEligibility.STALE_MAPPING: "Linked class missing from model",
    RecognitionEligibility.NO_CONSENT: "Linked – consent not recorded",
    RecognitionEligibility.INACTIVE: "Inactive",
}


def eligibility(student: dict[str, Any], model_labels: set[str] | None) -> RecognitionEligibility:
    if not int(student.get("is_active") or 0):
        return RecognitionEligibility.INACTIVE
    label = student.get("model_class_label")
    if not label or (isinstance(label, float)):
        return RecognitionEligibility.NOT_IN_MODEL
    if model_labels is None:
        return RecognitionEligibility.MODEL_UNAVAILABLE
    if label not in model_labels:
        return RecognitionEligibility.STALE_MAPPING
    consent = student.get("consent_recorded_at")
    if not consent or isinstance(consent, float):
        return RecognitionEligibility.NO_CONSENT
    return RecognitionEligibility.RECOGNIZABLE


def _natural_key(path: Path) -> tuple[int, str]:
    match = re.match(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def first_dataset_photo(dataset_dir: Path, label: str) -> Path | None:
    """First image of dataset/<label>/ (1.jpg before 10.jpg)."""
    folder = dataset_dir / label
    if not folder.is_dir():
        return None
    images = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key)
    return images[0] if images else None


def to_stored_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_photo(student: dict[str, Any], dataset_dir: Path | None = None) -> Path | None:
    """Photo file for a student: stored path, else the first training image of the linked class."""
    stored = student.get("photo_path")
    if isinstance(stored, str) and stored:
        path = Path(stored)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if path.is_file():
            return path
    label = student.get("model_class_label")
    if dataset_dir is not None and isinstance(label, str) and label:
        return first_dataset_photo(dataset_dir, label)
    return None


@dataclass(frozen=True)
class StudentForm:
    student_id: str
    full_name: str
    class_name: str
    section: str | None
    email: str | None
    model_class_label: str | None


class StudentService:
    def __init__(self, db: Database, dataset_dir: Path | None = None) -> None:
        self.db = db
        self.repo = StudentRepository(db)
        self.auth = AuthService(db)
        self.dataset_dir = dataset_dir

    def _validated(self, form: StudentForm, model_labels: set[str] | None) -> StudentInput:
        return validate_student(student_id=form.student_id, full_name=form.full_name, class_name=form.class_name,
                                section=form.section, email=form.email, model_class_label=form.model_class_label,
                                known_model_labels=model_labels)

    def register(self, form: StudentForm, *, consent: bool, model_labels: set[str] | None) -> int:
        """Create the student AND their login account (username = Student ID, demo password) together."""
        data = self._validated(form, model_labels)
        photo = (first_dataset_photo(self.dataset_dir, data.model_class_label)
                 if self.dataset_dir is not None and data.model_class_label else None)
        with self.db.transaction() as tx:
            pk = self.repo.create(data, consent=consent, photo_path=to_stored_path(photo), conn=tx)
            self.auth.create_student_login(student_pk=pk, username=data.student_id, display_name=data.full_name, conn=tx)
        return pk

    def update(self, pk: int, form: StudentForm, *, model_labels: set[str] | None) -> None:
        current = self.repo.get(pk)
        if current is None:
            raise ValueError("Student not found.")
        if form.model_class_label and form.model_class_label == current.get("model_class_label"):
            labels = (model_labels or set()) | {form.model_class_label}  # keep an existing link as-is
        else:
            labels = model_labels
        data = self._validated(form, labels)
        self.repo.update(pk, data)
        if data.model_class_label and data.model_class_label != current.get("model_class_label") and self.dataset_dir:
            self.repo.set_photo(pk, to_stored_path(first_dataset_photo(self.dataset_dir, data.model_class_label)))

    def set_active(self, pk: int, active: bool) -> None:
        self.repo.set_active(pk, active)

    def set_consent(self, pk: int, consent: bool) -> None:
        self.repo.set_consent(pk, consent)

    def get(self, pk: int) -> dict[str, Any] | None:
        return self.repo.get(pk)

    def photo(self, student: dict[str, Any]) -> Path | None:
        return resolve_photo(student, self.dataset_dir)

    def class_names(self) -> list[str]:
        return self.repo.class_names()

    def count_active(self) -> int:
        return self.repo.count(active_only=True)

    def all_students(self, *, active: bool | None = True) -> list[dict[str, Any]]:
        return self.repo.list(active=active)

    def directory(self, *, model_labels: set[str] | None, search: str | None = None, class_name: str | None = None,
                  active: bool | None = None) -> pd.DataFrame:
        rows = self.repo.list(search=search, class_name=class_name, active=active)
        columns = ["id", "student_id", "full_name", "class_name", "section", "email", "model_class_label",
                   "photo_path", "is_active", "consent_recorded_at", "created_at", "updated_at", "recognition_status"]
        if not rows:
            return pd.DataFrame(columns=columns)
        for row in rows:
            row["recognition_status"] = ELIGIBILITY_TEXT[eligibility(row, model_labels)]
        df = pd.DataFrame(rows)
        df["is_active"] = df["is_active"].astype(bool)
        return df[columns]

    def model_class_overview(self, class_names: tuple[str, ...]) -> pd.DataFrame:
        mapped = self.repo.mapped_labels()
        rows = []
        for index, label in enumerate(class_names):
            student = mapped.get(label)
            rows.append({
                "model_index": index,
                "model_class_label": label,
                "student_id": student["student_id"] if student else None,
                "full_name": student["full_name"] if student else None,
                "status": ELIGIBILITY_TEXT[eligibility(student, set(class_names))] if student else "Not linked to a student",
            })
        return pd.DataFrame(rows)

    def unlinked_labels(self, class_names: tuple[str, ...] | None) -> list[str]:
        if not class_names:
            return []
        mapped = self.repo.mapped_labels()
        return [label for label in class_names if label not in mapped]

    def stale_mappings(self, class_names: tuple[str, ...] | None) -> list[dict[str, Any]]:
        if class_names is None:
            return []
        known = set(class_names)
        return [row for label, row in self.repo.mapped_labels().items() if label not in known]

    def auto_link_matching_ids(self, class_names: tuple[str, ...]) -> list[tuple[str, str]]:
        """Explicit teacher action: link unlinked classes to unlinked students whose ID equals the class name."""
        mapped = self.repo.mapped_labels()
        created: list[tuple[str, str]] = []
        for label in class_names:
            if label in mapped:
                continue
            student = self.repo.get_by_student_id(label)
            if student is None or student.get("model_class_label"):
                continue
            self.repo.set_model_label(int(student["id"]), label)
            if self.dataset_dir is not None and not student.get("photo_path"):
                self.repo.set_photo(int(student["id"]), to_stored_path(first_dataset_photo(self.dataset_dir, label)))
            created.append((label, student["student_id"]))
        return created

    def overlay_names(self, class_names: tuple[str, ...]) -> dict[str, str]:
        """Model label -> student name, only for students the camera may recognise."""
        labels = set(class_names)
        return {label: row["full_name"] for label, row in self.repo.mapped_labels().items()
                if eligibility(row, labels) is RecognitionEligibility.RECOGNIZABLE}
