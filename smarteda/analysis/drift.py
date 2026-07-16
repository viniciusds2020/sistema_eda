"""Train/test data profiling and drift indicators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from smarteda.core.adapters import to_pandas


_EPS = 1e-6


def _psi(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    train = pd.to_numeric(train, errors="coerce").dropna()
    test = pd.to_numeric(test, errors="coerce").dropna()
    if train.empty or test.empty or train.nunique() < 2:
        return 0.0

    edges = np.unique(train.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    train_dist = (
        pd.cut(train, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
    )
    test_dist = (
        pd.cut(test, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
    )
    actual = np.clip(train_dist.to_numpy(dtype=float), _EPS, None)
    expected = np.clip(test_dist.reindex(train_dist.index, fill_value=0).to_numpy(dtype=float), _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _js_divergence(train: pd.Series, test: pd.Series) -> tuple[float, float]:
    train_values = train.astype("string").fillna("__MISSING__")
    test_values = test.astype("string").fillna("__MISSING__")
    categories = train_values.value_counts().index.union(test_values.value_counts().index)

    p = train_values.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    q = test_values.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    p = np.clip(p, _EPS, None)
    q = np.clip(q, _EPS, None)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))

    known = set(train_values.unique())
    unseen_ratio = float((~test_values.isin(known)).mean())
    return float(js), unseen_ratio


def _drift_level(score: float, *, numeric: bool) -> str:
    if numeric:
        if score >= 0.25:
            return "high"
        if score >= 0.10:
            return "medium"
    else:
        if score >= 0.20:
            return "high"
        if score >= 0.10:
            return "medium"
    return "low"


def profile_train_test(
    train: Any,
    test: Any,
    *,
    target: str | None = None,
    bins: int = 10,
    max_rows: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare schema, missingness, and distributions between train and test."""
    train_df = to_pandas(
        train,
        max_rows=max_rows,
        random_state=random_state,
    )
    test_df = to_pandas(
        test,
        max_rows=max_rows,
        random_state=random_state,
    )

    train_columns = set(train_df.columns)
    test_columns = set(test_df.columns)
    common = sorted(train_columns & test_columns)
    rows: list[dict[str, Any]] = []

    for column in common:
        if column == target:
            role = "target"
        else:
            role = "feature"

        train_series = train_df[column]
        test_series = test_df[column]
        numeric = (
            pd.api.types.is_numeric_dtype(train_series)
            and pd.api.types.is_numeric_dtype(test_series)
        )

        train_missing = float(train_series.isna().mean())
        test_missing = float(test_series.isna().mean())
        row: dict[str, Any] = {
            "column": column,
            "role": role,
            "kind": "numeric" if numeric else "categorical",
            "train_dtype": str(train_series.dtype),
            "test_dtype": str(test_series.dtype),
            "dtype_changed": str(train_series.dtype) != str(test_series.dtype),
            "train_missing_rate": train_missing,
            "test_missing_rate": test_missing,
            "missing_rate_delta": test_missing - train_missing,
            "train_unique": int(train_series.nunique(dropna=True)),
            "test_unique": int(test_series.nunique(dropna=True)),
        }

        if numeric:
            score = _psi(train_series, test_series, bins=bins)
            row.update(
                {
                    "metric": "psi",
                    "drift_score": score,
                    "drift_level": _drift_level(score, numeric=True),
                    "unseen_category_rate": None,
                    "train_mean": float(pd.to_numeric(train_series, errors="coerce").mean()),
                    "test_mean": float(pd.to_numeric(test_series, errors="coerce").mean()),
                }
            )
        else:
            score, unseen = _js_divergence(train_series, test_series)
            row.update(
                {
                    "metric": "js_divergence",
                    "drift_score": score,
                    "drift_level": _drift_level(score, numeric=False),
                    "unseen_category_rate": unseen,
                    "train_mean": None,
                    "test_mean": None,
                }
            )
        rows.append(row)

    high = sum(row["drift_level"] == "high" for row in rows)
    medium = sum(row["drift_level"] == "medium" for row in rows)

    return {
        "summary": {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "common_columns": len(common),
            "missing_in_test": sorted(train_columns - test_columns),
            "new_in_test": sorted(test_columns - train_columns),
            "high_drift_features": high,
            "medium_drift_features": medium,
            "max_rows_per_dataset": max_rows,
        },
        "features": rows,
        "thresholds": {
            "psi": {"medium": 0.10, "high": 0.25},
            "js_divergence": {"medium": 0.10, "high": 0.20},
        },
        "disclaimer": (
            "PSI and Jensen-Shannon thresholds are screening heuristics; "
            "validate them against model sensitivity and business impact."
        ),
    }


def profile_frame(comparison: dict[str, Any]) -> pd.DataFrame:
    """Return the feature comparison as a DataFrame."""
    return pd.DataFrame(comparison.get("features", []))
