"""Evidence-based preprocessing diagnostics with optional target awareness."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _fdr_bh(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values in their original order."""
    if not pvalues:
        return []
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result.tolist()


def _target_type(series: pd.Series, classification_threshold: int) -> str:
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    return (
        "classification"
        if series.nunique(dropna=True) <= classification_threshold
        else "regression"
    )


def _missing_target_pvalue(
    missing: pd.Series, target: pd.Series, target_type: str
) -> Tuple[Optional[str], Optional[float]]:
    valid = target.notna()
    missing = missing[valid]
    target = target[valid]
    if missing.nunique() < 2 or len(target) < 10:
        return None, None
    try:
        if target_type == "classification":
            table = pd.crosstab(missing, target)
            if min(table.shape) < 2:
                return None, None
            return "chi_square_missingness_vs_target", float(stats.chi2_contingency(table)[1])
        present_target = target[~missing]
        missing_target = target[missing]
        if min(len(present_target), len(missing_target)) < 3:
            return None, None
        return "mann_whitney_missingness_vs_target", float(
            stats.mannwhitneyu(present_target, missing_target, alternative="two-sided").pvalue
        )
    except (TypeError, ValueError):
        return None, None


def preprocessing_diagnostics(
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    classification_threshold: int = 10,
    alpha: float = 0.05,
    max_normality_sample: int = 5000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Recommend preprocessing from aggregate evidence without changing data.

    Suggestions are diagnostics rather than transformations. Fitting imputers,
    encoders and scalers must happen inside the training fold to avoid leakage.
    """
    target_series = df[target] if target and target in df.columns else None
    target_kind = (
        _target_type(target_series, classification_threshold) if target_series is not None else None
    )
    missing_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    normality_rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        missing_rate = float(series.isna().mean())
        if missing_rate > 0:
            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna().astype(float)
                skewness = float(clean.skew()) if len(clean) > 2 else 0.0
                strategy = "median" if abs(skewness) > 1 else "mean_or_median_cv"
            else:
                strategy = "explicit_missing_category_or_mode_cv"
            test, pvalue = (None, None)
            if target_series is not None:
                test, pvalue = _missing_target_pvalue(
                    series.isna(), target_series, str(target_kind)
                )
            missing_rows.append(
                {
                    "column": str(column),
                    "missing_rate": missing_rate,
                    "suggested_strategy": strategy,
                    "target_test": test,
                    "target_pvalue": pvalue,
                }
            )

        if not pd.api.types.is_numeric_dtype(series):
            continue
        clean = series.dropna().astype(float)
        if len(clean) < 3 or clean.nunique() < 2:
            continue
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr > 0:
            mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
            outlier_rate = float(mask.mean())
            if outlier_rate > 0:
                outlier_rows.append(
                    {
                        "column": str(column),
                        "method": "iqr_1.5",
                        "outlier_rate": outlier_rate,
                        "suggested_action": (
                            "investigate_then_robust_scale_or_winsorize"
                            if outlier_rate >= 0.01
                            else "review_extreme_observations"
                        ),
                        "warning": "Do not remove observations before checking target impact.",
                    }
                )
        sample = clean
        sampled = len(clean) > max_normality_sample
        if sampled:
            sample = clean.sample(max_normality_sample, random_state=random_state)
        try:
            statistic, pvalue = stats.shapiro(sample)
            is_normal = bool(pvalue >= alpha)
            skewness = float(clean.skew())
            if is_normal:
                suggestion = "standard_scaler_if_required_by_model"
            elif clean.min() >= 0 and abs(skewness) > 1:
                suggestion = "consider_log1p_or_yeo_johnson_then_validate"
            else:
                suggestion = "consider_yeo_johnson_or_robust_scaler_then_validate"
            normality_rows.append(
                {
                    "column": str(column),
                    "test": "shapiro_wilk",
                    "sample_size": len(sample),
                    "sampled": sampled,
                    "statistic": float(statistic),
                    "pvalue": float(pvalue),
                    "is_normal": is_normal,
                    "skewness": skewness,
                    "suggested_action": suggestion,
                }
            )
        except (TypeError, ValueError):
            pass

    tested_missing = [row for row in missing_rows if row["target_pvalue"] is not None]
    adjusted = _fdr_bh([float(row["target_pvalue"]) for row in tested_missing])
    for row, adjusted_pvalue in zip(tested_missing, adjusted):
        row["target_adjusted_pvalue"] = adjusted_pvalue
        row["target_associated"] = adjusted_pvalue < alpha
        if row["target_associated"]:
            actions.append(
                {
                    "priority": "high",
                    "column": row["column"],
                    "action": "model_missingness_indicator_and_validate_by_target",
                    "evidence": f"missingness adjusted p-value={adjusted_pvalue:.4g}",
                }
            )

    if missing_rows:
        actions.append(
            {
                "priority": "medium",
                "column": None,
                "action": "fit_imputation_inside_each_training_fold",
                "evidence": f"{len(missing_rows)} columns contain missing values",
            }
        )
    high_outliers = [row for row in outlier_rows if row["outlier_rate"] >= 0.01]
    if high_outliers:
        actions.append(
            {
                "priority": "medium",
                "column": None,
                "action": "validate_outliers_by_target_before_robust_transformation",
                "evidence": f"{len(high_outliers)} columns have at least 1% IQR outliers",
            }
        )

    target_actions: list[dict[str, Any]] = []
    if target_series is not None:
        target_missing = float(target_series.isna().mean())
        if target_missing > 0:
            target_actions.append(
                {
                    "priority": "high",
                    "action": "exclude_unlabeled_rows_from_supervised_fit_or_define_labeling_policy",
                    "evidence": f"target missing rate={target_missing:.1%}",
                }
            )
        if target_kind == "classification":
            counts = target_series.value_counts(dropna=True)
            balance = float(counts.min() / counts.max()) if len(counts) > 1 else 0.0
            if balance < 0.5:
                target_actions.append(
                    {
                        "priority": "high",
                        "action": "use_stratified_split_and_class_weight_or_resampling_inside_cv",
                        "evidence": f"minority/majority ratio={balance:.3f}",
                    }
                )
        else:
            clean_target = target_series.dropna().astype(float)
            if len(clean_target) >= 3 and abs(float(clean_target.skew())) > 1:
                target_actions.append(
                    {
                        "priority": "medium",
                        "action": "evaluate_target_transformation_with_metrics_on_original_scale",
                        "evidence": f"target skewness={clean_target.skew():.3f}",
                    }
                )
    actions.extend(target_actions)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    actions.sort(key=lambda row: priority_order.get(str(row["priority"]), 9))
    return {
        "summary": {
            "columns_with_missing": len(missing_rows),
            "columns_with_outliers": len(outlier_rows),
            "non_normal_numeric_columns": sum(not row["is_normal"] for row in normality_rows),
            "target_type": target_kind,
            "prioritized_actions": len(actions),
        },
        "missing_data": missing_rows,
        "outliers": outlier_rows,
        "normality_tests": normality_rows,
        "target_actions": target_actions,
        "prioritized_actions": actions,
        "disclaimer": (
            "Recommendations are diagnostic. Fit preprocessing only on training folds "
            "and validate choices with domain knowledge and model metrics."
        ),
    }
