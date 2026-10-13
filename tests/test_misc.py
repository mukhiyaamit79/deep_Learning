import math

import pandas as pd
import pytest

from services.reporting_service import aggregate, attendance_rate
from utils.security import hash_password, verify_password
from utils.validators import ValidationError, validate_session, validate_student


def test_password_hashing():
    encoded = hash_password("teacher123", iterations=1000)
    assert verify_password("teacher123", encoded)
    assert not verify_password("wrong", encoded)
    assert "teacher123" not in encoded


def test_attendance_rate_definition():
    assert attendance_rate(3, 1) == 0.75
    assert math.isnan(attendance_rate(0, 0))
    df = pd.DataFrame({"session_id": [1, 1, 1, 2, 2], "g": ["A"] * 5,
                       "status": ["present", "late", "absent", "excused", "present"]})
    row = aggregate(df, ["g"]).iloc[0]
    assert row["sessions"] == 2 and row["attendance_rate"] == 0.75


def test_validation():
    assert validate_student(student_id=" Amit ", full_name=" Amit ", class_name="TECH 405").student_id == "Amit"
    with pytest.raises(ValidationError):
        validate_student(student_id="has space", full_name="A", class_name="B")
    with pytest.raises(ValidationError):
        validate_student(student_id="S1", full_name="A", class_name="B", model_class_label="X", known_model_labels={"Y"})
    with pytest.raises(ValidationError):
        validate_session(session_name="L", session_date="2026-10-11", class_name="C", start_time="10:00",
                         end_time="09:00")
