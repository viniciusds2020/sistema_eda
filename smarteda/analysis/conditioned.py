"""Target-conditioned drift profiling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from smarteda.analysis.drift import profile_train_test
from smarteda.core.adapters import to_pandas


def _target_segments(
    train_target: pd.Series,
    test_target: pd.Series,
    *,
    target_bins: int,
    classification_threshold: int,
) -> tuple[pd.Series, pd.Series, str]:
    combined_unique = pd.concat([train_target, test_target]).nunique(dropna=True)
    if (
        not pd.api.types.is_numeric_dtype(train_target)
        or combined_unique <= classification_threshold
    ):
        return (
            train_target.astype("string").fillna("__MISSING__"),
            test_target.astype("string").fillna("__MISSING__"),
            "classes",
        )

    numeric_train = pd.to_numeric(train_target, errors="coerce")
    unique = numeric_train.dropna().nunique()
    quantiles = min(target_bins, unique)
    if quantiles < 2:
        return (
            train_target.astype("string").fillna("__MISSING__"),
            test_target.astype("string").fillna("__MISSING__"),
            "classes",
        )

    edges = np.unique(
        numeric_train.dropna().quantile(np.linspace(0, 1, quantiles + 1)).to_numpy()
    )
    if len(edges) < 3:
        return (
            train_target.astype("string").fillna("__MISSING__"),
            test_target.astype("string").fillna("__MISSING__"),
            "classes",
        )
    edges[0], edges[-1] = -np.inf, np.inf
    train_segment = pd.cut(numeric_train, bins=edges, include_lowest=True).astype(str)
    test_segment = pd.cut(
        pd.to_numeric(test_target, errors="coerce"),
        bins=edges,
        include_lowest=True,
    ).astype(str)
    return train_segment, test_segment, "train_quantiles"


def target_conditioned_drift(
    train: Any,
    test: Any,
    *,
    target: str,
    bins: int = 10,
    target_bins: int = 5,
    classification_threshold: int = 10,
    min_samples: int = 30,
    max_rows: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Measure feature drift inside target classes or target quantile bands."""
    train_df = to_pandas(train, max_rows=max_rows, random_state=random_state)
    test_df = to_pandas(test, max_rows=max_rows, random_state=random_state)
    if target not in train_df or target not in test_df:
        raise ValueError(f"Target '{target}' must exist in train and test.")

    train_segment, test_segment, strategy = _target_segments(
        train_df[target],
        test_df[target],
        target_bins=target_bins,
        classification_threshold=classification_threshold,
    )
    segments = sorted(set(train_segment.dropna()) | set(test_segment.dropna()))
    feature_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    feature_columns = sorted((set(train_df.columns) & set(test_df.columns)) - {target})
    for segment in segments:
        train_mask = train_segment == segment
        test_mask = test_segment == segment
        train_count = int(train_mask.sum())
        test_count = int(test_mask.sum())
        status = "compared" if min(train_count, test_count) >= min_samples else "insufficient_samples"

        segment_rows.append(
            {
                "target_segment": str(segment),
                "train_count": train_count,
                "test_count": test_count,
                "status": status,
            }
        )
        if status != "compared":
            continue

        comparison = profile_train_test(
            train_df.loc[train_mask, feature_columns],
            test_df.loc[test_mask, feature_columns],
            bins=bins,
        )
        for row in comparison["features"]:
            feature_rows.append(
                {
                    **row,
                    "target_segment": str(segment),
                    "segment_train_count": train_count,
                    "segment_test_count": test_count,
                }
            )

    return {
        "summary": {
            "target": target,
            "segmentation_strategy": strategy,
            "segments_total": len(segment_rows),
            "segments_compared": sum(row["status"] == "compared" for row in segment_rows),
            "segments_skipped": sum(row["status"] != "compared" for row in segment_rows),
            "high_drift_findings": sum(row["drift_level"] == "high" for row in feature_rows),
            "min_samples": min_samples,
        },
        "segments": segment_rows,
        "features": feature_rows,
        "disclaimer": (
            "Conditioning on the target is diagnostic. It can reveal conditional drift "
            "but does not replace model-performance monitoring."
        ),
    }


def conditioned_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(result.get("features", []))
