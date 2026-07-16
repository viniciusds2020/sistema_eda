"""Longitudinal drift monitoring across multiple data windows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from smarteda.analysis.drift import profile_train_test


def monitor_windows(
    reference: Any,
    windows: Mapping[str, Any],
    *,
    target: str | None = None,
    bins: int = 10,
    max_rows: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare multiple named windows against a fixed reference dataset."""
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, Any]] = {}

    for order, (window_name, frame) in enumerate(windows.items()):
        profile = profile_train_test(
            reference,
            frame,
            target=target,
            bins=bins,
            max_rows=max_rows,
            random_state=random_state,
        )
        profiles[str(window_name)] = profile
        summary = profile["summary"]
        summaries.append(
            {
                "window": str(window_name),
                "window_order": order,
                "rows": summary["test_rows"],
                "high_drift_features": summary["high_drift_features"],
                "medium_drift_features": summary["medium_drift_features"],
                "new_columns": len(summary["new_in_test"]),
                "missing_columns": len(summary["missing_in_test"]),
            }
        )
        for row in profile["features"]:
            history.append(
                {
                    **row,
                    "window": str(window_name),
                    "window_order": order,
                }
            )

    return {
        "summary": {
            "windows": len(summaries),
            "max_rows_per_window": max_rows,
            "windows_with_high_drift": sum(row["high_drift_features"] > 0 for row in summaries),
        },
        "windows": summaries,
        "feature_history": history,
        "profiles": profiles,
        "disclaimer": (
            "All windows are compared with the same reference. "
            "Use stable window definitions and investigate persistent trends."
        ),
    }


def longitudinal_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(result.get("feature_history", []))
