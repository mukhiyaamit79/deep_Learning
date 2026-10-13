"""Image helpers for face detection, cropping and drawing.

The CNN input preprocessing is NOT defined here. It is the notebook's own function
``attendance_utils.preprocess_image`` (Pillow grayscale "L", bilinear resize to 128×128,
float32 / 255), which the recognition service calls directly — so the web app feeds the model
exactly what it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


class InvalidImageError(ValueError):
    """Raised when an image cannot be decoded or has an unsupported shape."""


# Overlay colours (OpenCV uses BGR order).
COLOR_RECOGNIZED = (60, 160, 40)   # green
COLOR_CANDIDATE = (200, 140, 20)   # blue: candidate, still verifying
COLOR_UNCERTAIN = (0, 165, 255)    # amber
COLOR_REJECTED = (40, 40, 220)     # red


@dataclass(frozen=True)
class FaceBox:
    """Axis-aligned face bounding box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return max(self.w, 0) * max(self.h, 0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def box_iou(first: FaceBox | None, second: FaceBox | None) -> float:
    """Intersection-over-union of two boxes (same rule as live_attendance.box_iou)."""
    if first is None or second is None:
        return 0.0
    ix = max(0, min(first.x + first.w, second.x + second.w) - max(first.x, second.x))
    iy = max(0, min(first.y + first.h, second.y + second.h) - max(first.y, second.y))
    intersection = ix * iy
    return intersection / max(1, first.area + second.area - intersection)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG/... bytes into a BGR uint8 array. Raises InvalidImageError."""
    if not data:
        raise InvalidImageError("The image is empty.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidImageError("The file could not be decoded as an image (damaged or unsupported format).")
    return image


def validate_image_array(image: Any) -> np.ndarray:
    """Return ``image`` as uint8 with 2 or 3 dimensions, or raise InvalidImageError."""
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("Expected a NumPy image array.")
    if image.size == 0 or image.ndim not in (2, 3):
        raise InvalidImageError(f"Unsupported image shape {getattr(image, 'shape', None)}.")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise InvalidImageError(f"Unsupported number of channels: {image.shape[2]}.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR / BGRA / single-channel -> 2-D uint8 grayscale (used for face DETECTION only)."""
    image = validate_image_array(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    image = validate_image_array(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """OpenCV BGR crop -> Pillow RGB image (the input type of attendance_utils.preprocess_image)."""
    return Image.fromarray(cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2RGB))


def expand_box(box: FaceBox, margin: float, image_shape: tuple[int, ...]) -> FaceBox:
    """Grow ``box`` by ``margin`` (fraction of its size) on every side, clipped to the image."""
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    dx, dy = int(round(box.w * margin)), int(round(box.h * margin))
    x0, y0 = max(box.x - dx, 0), max(box.y - dy, 0)
    x1, y1 = min(box.x + box.w + dx, img_w), min(box.y + box.h + dy, img_h)
    return FaceBox(x0, y0, max(x1 - x0, 0), max(y1 - y0, 0))


def shift_box(box: FaceBox, dx_frac: float, dy_frac: float, scale: float, image_shape: tuple[int, ...]) -> FaceBox:
    """Slightly shifted / scaled copy of ``box`` (used for still-photo crop variants)."""
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    new_w, new_h = max(int(round(box.w * scale)), 1), max(int(round(box.h * scale)), 1)
    cx = box.x + box.w / 2.0 + dx_frac * box.w
    cy = box.y + box.h / 2.0 + dy_frac * box.h
    x0 = min(max(int(round(cx - new_w / 2.0)), 0), max(img_w - 1, 0))
    y0 = min(max(int(round(cy - new_h / 2.0)), 0), max(img_h - 1, 0))
    return FaceBox(x0, y0, min(new_w, img_w - x0), min(new_h, img_h - y0))


def crop_box(image: np.ndarray, box: FaceBox) -> np.ndarray:
    if box.w <= 1 or box.h <= 1:
        raise InvalidImageError("The face crop is empty.")
    return image[box.y : box.y + box.h, box.x : box.x + box.w]


def crop_face(image: np.ndarray, box: FaceBox, margin: float = 0.0) -> np.ndarray:
    """Crop ``box`` expanded by ``margin``. With margin 0 this equals the notebook live script's crop."""
    return crop_box(image, expand_box(box, margin, image.shape))


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def draw_face_box(image: np.ndarray, box: FaceBox, label: str | None, color: tuple[int, int, int],
                  thickness: int = 2) -> np.ndarray:
    """Draw a rectangle and an optional caption (in place; returns the image)."""
    cv2.rectangle(image, (box.x, box.y), (box.x + box.w, box.y + box.h), color, thickness)
    if label:
        text = _ascii(label)
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
        (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
        top = max(box.y - th - baseline - 6, 0)
        cv2.rectangle(image, (box.x, top), (box.x + tw + 8, top + th + baseline + 6), color, cv2.FILLED)
        cv2.putText(image, text, (box.x + 4, top + th + 2), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def draw_banner(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Status banner across the top-left of a frame (in place)."""
    text = _ascii(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, 0.6, 1)
    cv2.rectangle(image, (0, 0), (min(tw + 16, image.shape[1]), th + baseline + 12), color, cv2.FILLED)
    cv2.putText(image, text, (8, th + 6), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image
