"""Dataset quality diagnostics and lightweight leakage heuristics."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


_ID_NAME = re.compile(r"(^id$|^id_|_id$|uuid|guid|identifier|codigo$|^codigo_)", re.I)
_LEAKAGE_NAME = re.compile(
    r"(target|label|outcome|resultado|prediction|predicted|proba|probability|score_final)",
    re.I,
)


def detect_quality_issues(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    id_unique_ratio: float = 0.95,
    leakage_corr_threshold: float = 0.999,
) -> dict[str, Any]:
    """Detect constants, possible identifiers, and potential target leakage.

    The leakage checks are warnings, not proof. They combine suspicious names,
    exact target copies, and near-perfect numeric correlation.
    """
    n_rows = len(df)
    constants: list[dict[str, Any]] = []
    possible_ids: list[dict[str, Any]] = []
    leakage: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        unique_ratio = float(unique_count / max(len(non_null), 1))

        if unique_count <= 1:
            constants.append(
                {
                    "column": column,
                    "unique_count": unique_count,
                    "reason": "constant_or_all_missing",
                    "severity": "high",
                }
            )

        name_hint = bool(_ID_NAME.search(str(column)))
        integer_sequence = (
            pd.api.types.is_integer_dtype(series)
            and len(non_null) > 2
            and (non_null.is_monotonic_increasing or non_null.is_monotonic_decreasing)
        )
        if n_rows > 0 and unique_ratio >= id_unique_ratio and (name_hint or integer_sequence):
            possible_ids.append(
                {
                    "column": column,
                    "unique_ratio": unique_ratio,
                    "reason": "name_and_cardinality" if name_hint else "monotonic_unique_integer",
                    "severity": "medium",
                }
            )

    if target and target in df.columns:
        target_series = df[target]
        for column in df.columns:
            if column == target:
                continue

            series = df[column]
            if _LEAKAGE_NAME.search(str(column)):
                leakage.append(
                    {
                        "column": column,
                        "check": "suspicious_name",
                        "value": str(column),
                        "severity": "medium",
                    }
                )

            valid = series.notna() & target_series.notna()
            if not valid.any():
                continue

            left = series[valid].reset_index(drop=True)
            right = target_series[valid].reset_index(drop=True)
            if left.astype(str).equals(right.astype(str)):
                leakage.append(
                    {
                        "column": column,
                        "check": "exact_target_copy",
                        "value": 1.0,
                        "severity": "high",
                    }
                )
                continue

            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                if left.nunique() > 1 and right.nunique() > 1:
                    correlation = float(np.corrcoef(left.astype(float), right.astype(float))[0, 1])
                    if np.isfinite(correlation) and abs(correlation) >= leakage_corr_threshold:
                        leakage.append(
                            {
                                "column": column,
                                "check": "near_perfect_target_correlation",
                                "value": correlation,
                                "severity": "high",
                            }
                        )

    duplicated_rows = int(df.duplicated().sum())
    missing_by_column = {
        str(column): float(df[column].isna().mean()) for column in df.columns
    }

    return {
        "summary": {
            "rows": n_rows,
            "columns": len(df.columns),
            "constant_columns": len(constants),
            "possible_id_columns": len(possible_ids),
            "possible_leakage_warnings": len(leakage),
            "duplicated_rows": duplicated_rows,
        },
        "constant_columns": constants,
        "possible_ids": possible_ids,
        "possible_leakage": leakage,
        "missing_by_column": missing_by_column,
        "disclaimer": (
            "Leakage and identifier checks are heuristic warnings and require domain validation."
        ),
    }
