"""Shared inference preprocessing and saved-bundle loading; no training on import."""
from attendance_runtime import configure_runtime
configure_runtime()

import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESSING = {
    "height": 128, "width": 128, "channels": 1,
    "color_mode": "L", "resize": "bilinear", "normalization_divisor": 255.0,
    "dtype": "float32", "input_layout": "NCHW", "preprocessed_array_layout": "HWC",
    "confidence_threshold": 0.90, "consistent_frames": 8,
    "threshold_status": "Predefined demonstration heuristic; not calibrated or open-set validated",
}


def preprocess_image(image, config=None):
    """Pillow L conversion + bilinear resize, exactly as the notebook loader."""
    config = DEFAULT_PREPROCESSING if config is None else config
    if not isinstance(image, Image.Image):
        with Image.open(image) as opened:
            return preprocess_image(opened, config)
    resized = image.convert("L").resize(
        (int(config["width"]), int(config["height"])), Image.Resampling.BILINEAR)
    return (np.asarray(resized, dtype=np.float32) / 255.0)[..., np.newaxis]


def load_bundle(model_dir=None):
    """Load the trusted, locally trained model and validate its inference contract."""
    import torch
    from attendance_models import AttendanceCNN, choose_device

    directory = Path(model_dir) if model_dir else PROJECT_ROOT / "models"
    required = ["attendance_cnn.pt", "class_names.json", "preprocessing_config.json", "model_metadata.json"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)} in {directory}. Run notebook NN-36 first.")
    names = json.loads((directory / "class_names.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "preprocessing_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError("Preprocessing configuration and model metadata must be JSON objects.")
    if metadata.get("framework") != "pytorch" or metadata.get("format_version") != 1:
        raise ValueError("This is not a supported PyTorch bundle. Retrain and run NN-36; Keras files are incompatible.")
    if (not isinstance(names, list) or len(names) < 2
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Class mapping must be an ordered list of unique student IDs.")
    for key in ("height", "width", "channels", "color_mode", "resize",
                "normalization_divisor", "dtype", "input_layout"):
        if config.get(key) != DEFAULT_PREPROCESSING[key]:
            raise ValueError(f"Unsupported preprocessing setting: {key}={config.get(key)!r}")
    try:
        threshold = float(config["confidence_threshold"])
        frame_count = int(config["consistent_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid confidence policy in preprocessing_config.json.") from exc
    if not 0 < threshold <= 1 or frame_count < 2:
        raise ValueError("Invalid confidence threshold or consistent-frame count.")
    config["confidence_threshold"] = threshold
    config["consistent_frames"] = frame_count
    architecture = metadata.get("architecture_config")
    if not isinstance(architecture, dict) or architecture.get("num_classes") != len(names):
        raise ValueError("Model architecture and class mapping do not agree.")
    model = AttendanceCNN(**architecture)
    state = torch.load(directory / "attendance_cnn.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    # Match the notebook's float32 inference policy rather than enabling TF32 implicitly.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model.to(choose_device()).eval()
    return model, names, config


def predict_face(model, names, config, image, threshold=None):
    """Return an uncalibrated softmax score and heuristic rejection decision."""
    import torch

    threshold = config["confidence_threshold"] if threshold is None else threshold
    if not 0 < float(threshold) <= 1:
        raise ValueError("Threshold must be in (0, 1].")
    device = next(model.parameters()).device
    tensor = torch.from_numpy(preprocess_image(image, config)).permute(2, 0, 1).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        scores = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    if scores.shape != (len(names),) or not np.isfinite(scores).all():
        raise ValueError("Model returned invalid class scores.")
    index = int(np.argmax(scores))
    confidence = float(scores[index])
    return {"student_id": names[index] if confidence >= threshold else "Unknown",
            "candidate": names[index], "confidence": confidence,
            "accepted": confidence >= threshold,
            "top_classes": [(names[int(i)], float(scores[i]))
                            for i in np.argsort(scores)[::-1][:min(3, len(names))]]}
