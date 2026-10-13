"""Read the experiment results exported by amit_eda.ipynb (NN-39) from results/.

The app never trains models and never invents metrics. Missing files are reported as missing.
Only results/figures/ is used for images, so unrelated files elsewhere in results/ are ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import COMPARISON_REQUIRED_COLUMNS, FIGURE_TITLES, AppPaths


@dataclass
class ComparisonResults:
    comparison: pd.DataFrame | None = None
    per_class: pd.DataFrame | None = None
    experiments: pd.DataFrame | None = None
    validation: pd.DataFrame | None = None
    summary: dict[str, Any] | None = None
    figures: dict[str, Path] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.comparison is not None and not self.comparison.empty


def _read_csv(path: Path, result: ComparisonResults, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    if not path.is_file():
        result.missing.append(path.name)
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        result.errors.append(f"{path.name} could not be read: {exc}")
        return None
    absent = [c for c in required if c not in df.columns]
    if absent:
        result.errors.append(f"{path.name} is missing columns: {', '.join(absent)}. Re-run notebook NN-39.")
        return None
    return df


def load_results(paths: AppPaths) -> ComparisonResults:
    result = ComparisonResults()
    result.comparison = _read_csv(paths.comparison_csv, result, COMPARISON_REQUIRED_COLUMNS)
    result.per_class = _read_csv(paths.per_class_csv, result, ("Model", "Student ID", "Precision", "Recall", "F1-score"))
    result.experiments = _read_csv(paths.experiments_csv, result, ("Model", "Experiment"))
    result.validation = _read_csv(paths.validation_csv, result, ("Model",))
    if paths.summary_json.is_file():
        try:
            data = json.loads(paths.summary_json.read_text(encoding="utf-8"))
            result.summary = data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError) as exc:
            result.errors.append(f"summary.json could not be read: {exc}")
    else:
        result.missing.append(paths.summary_json.name)
    for name in FIGURE_TITLES:
        path = paths.figures_dir / name
        if path.is_file():
            result.figures[name] = path
    return result


def best_model(comparison: pd.DataFrame | None, metric: str) -> list[tuple[str, float]]:
    """All models tied for the highest value of ``metric`` (ties are reported, not hidden)."""
    if comparison is None or comparison.empty or metric not in comparison.columns:
        return []
    values = pd.to_numeric(comparison[metric], errors="coerce")
    if values.isna().all():
        return []
    top = values.max()
    return [(str(m), float(v)) for m, v in zip(comparison["Model"], values) if v == top]
