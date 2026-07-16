"""Distribution tests with multiple-comparison correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from smarteda.core.adapters import to_pandas


def adjust_pvalues(
    pvalues: list[float],
    *,
    method: str = "fdr_bh",
) -> np.ndarray:
    """Adjust p-values using Benjamini-Hochberg FDR or Bonferroni."""
    values = np.asarray(pvalues, dtype=float)
    if values.size == 0:
        return values

    method = method.lower()
    if method == "bonferroni":
        return np.minimum(values * len(values), 1.0)
    if method not in {"fdr_bh", "benjamini-hochberg", "bh"}:
        raise ValueError("correction must be 'fdr_bh' or 'bonferroni'.")

    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def _numeric_test(train: pd.Series, test: pd.Series) -> tuple[float, float, int, int]:
    left = pd.to_numeric(train, errors="coerce").dropna()
    right = pd.to_numeric(test, errors="coerce").dropna()
    if left.empty or right.empty:
        return 0.0, 1.0, len(left), len(right)
    result = stats.ks_2samp(left, right, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue), len(left), len(right)


def _categorical_test(
    train: pd.Series,
    test: pd.Series,
) -> tuple[float, float, int, int]:
    left = train.astype("string").fillna("__MISSING__")
    right = test.astype("string").fillna("__MISSING__")
    categories = left.value_counts().index.union(right.value_counts().index)
    table = np.vstack(
        [
            left.value_counts().reindex(categories, fill_value=0).to_numpy(),
            right.value_counts().reindex(categories, fill_value=0).to_numpy(),
        ]
    )
    nonzero = table.sum(axis=0) > 0
    table = table[:, nonzero]
    if table.shape[1] < 2:
        return 0.0, 1.0, len(left), len(right)

    chi2, pvalue, _, _ = stats.chi2_contingency(table)
    total = table.sum()
    effect = np.sqrt(chi2 / total) if total else 0.0
    return float(effect), float(pvalue), len(left), len(right)


def distribution_tests(
    train: Any,
    test: Any,
    *,
    target: str | None = None,
    alpha: float = 0.05,
    correction: str = "fdr_bh",
    max_rows: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Test train/test distribution changes feature-by-feature.

    Numeric variables use the two-sample Kolmogorov-Smirnov test. Categorical
    variables use a chi-square homogeneity test. P-values are adjusted across
    all tested features.
    """
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
    common = sorted(set(train_df.columns) & set(test_df.columns))
    rows: list[dict[str, Any]] = []

    for column in common:
        if column == target:
            continue
        left = train_df[column]
        right = test_df[column]
        numeric = (
            pd.api.types.is_numeric_dtype(left)
            and pd.api.types.is_numeric_dtype(right)
        )
        if numeric:
            effect, pvalue, n_train, n_test = _numeric_test(left, right)
            test_name = "kolmogorov_smirnov"
            effect_name = "ks_statistic"
        else:
            effect, pvalue, n_train, n_test = _categorical_test(left, right)
            test_name = "chi_square_homogeneity"
            effect_name = "cramers_v"

        rows.append(
            {
                "column": column,
                "kind": "numeric" if numeric else "categorical",
                "test": test_name,
                "effect_name": effect_name,
                "effect_size": effect,
                "pvalue": pvalue,
                "n_train": n_train,
                "n_test": n_test,
            }
        )

    adjusted = adjust_pvalues(
        [row["pvalue"] for row in rows],
        method=correction,
    )
    for row, adjusted_pvalue in zip(rows, adjusted):
        row["adjusted_pvalue"] = float(adjusted_pvalue)
        row["significant"] = bool(adjusted_pvalue < alpha)

    return {
        "summary": {
            "features_tested": len(rows),
            "significant_after_correction": sum(row["significant"] for row in rows),
            "alpha": alpha,
            "correction": correction,
            "max_rows_per_dataset": max_rows,
        },
        "features": rows,
        "disclaimer": (
            "Statistical significance is sample-size sensitive. "
            "Interpret adjusted p-values together with effect sizes."
        ),
    }


def tests_frame(result: dict[str, Any]) -> pd.DataFrame:
    """Return test results as a DataFrame."""
    return pd.DataFrame(result.get("features", []))
