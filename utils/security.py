"""Password hashing for the demo login (salted PBKDF2-HMAC-SHA256, Python standard library only).

Plain-text passwords are never stored. The demo passwords from utils/config.py are hashed when
the accounts are created. This is suitable for a local classroom demo, not for production.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 150_000


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_s, salt_hex, digest_hex = encoded.split("$")
        if algorithm != ALGORITHM:
            return False
        iterations = int(iterations_s)
        salt, expected = bytes.fromhex(salt_hex), bytes.fromhex(digest_hex)
    except (AttributeError, ValueError):
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def dummy_verify() -> None:
    """Spend similar time when a username does not exist (does not reveal which usernames exist)."""
    hashlib.pbkdf2_hmac("sha256", b"x", b"\x00" * 16, ITERATIONS)
