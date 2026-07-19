"""Build a compact, JSON-safe context without materializing dataset rows."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class InsightContextBuilder:
    """Select and compact statistical outputs before an external LLM call."""

    SECTIONS = (
        "dataset_name",
        "overview",
        "quality_diagnostics",
        "train_test_profile",
        "statistical_drift_tests",
        "target_conditioned_drift",
        "longitudinal_monitoring",
        "preprocessing_diagnostics",
        "time_series_diagnostics",
    )

    def __init__(self, max_items: int = 30, max_text_length: int = 160):
        self.max_items = max_items
        self.max_text_length = max_text_length

    def build(self, results: dict[str, Any]) -> dict[str, Any]:
        """Return aggregate statistics only; raw DataFrames are never accepted."""
        return {key: self._compact(results[key]) for key in self.SECTIONS if key in results}

    def _compact(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return round(value, 6) if np.isfinite(value) else None
        if isinstance(value, str):
            cleaned = " ".join(value.replace("<", "").replace(">", "").split())
            return cleaned[: self.max_text_length]
        if isinstance(value, Enum):
            return self._compact(value.value)
        if is_dataclass(value):
            return self._compact(asdict(value))
        if isinstance(value, pd.DataFrame):
            raise ValueError("Raw DataFrames are not allowed in the insight context.")
        if isinstance(value, dict):
            return {
                self._compact(str(key)): self._compact(item)
                for key, item in list(value.items())[: self.max_items]
            }
        if isinstance(value, (list, tuple)):
            return [self._compact(item) for item in value[: self.max_items]]
        if isinstance(value, np.generic):
            return self._compact(value.item())
        return self._compact(str(value))
