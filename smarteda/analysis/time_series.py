"""Time-series diagnostics and leakage-safe preprocessing recommendations."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


def _seasonal_period(frequency: Optional[str]) -> Optional[int]:
    if not frequency:
        return None
    value = frequency.upper()
    if value.startswith(("H", "60MIN")):
        return 24
    if value.startswith(("D", "B")):
        return 7
    if value.startswith("W"):
        return 52
    if value.startswith(("M", "ME", "MS")):
        return 12
    if value.startswith(("Q", "QE", "QS")):
        return 4
    return None


def _adf(values: pd.Series) -> dict[str, Any]:
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {
            "test": "adf",
            "available": False,
            "pvalue": None,
            "stationary": None,
            "note": "Install smarteda[timeseries] for the Augmented Dickey-Fuller test.",
        }
    try:
        statistic, pvalue, used_lag, nobs, _, _ = adfuller(values.astype(float), autolag="AIC")
        return {
            "test": "adf",
            "available": True,
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "used_lag": int(used_lag),
            "nobs": int(nobs),
            "stationary": bool(pvalue < 0.05),
        }
    except (ValueError, np.linalg.LinAlgError):
        return {
            "test": "adf",
            "available": True,
            "pvalue": None,
            "stationary": None,
            "note": "Insufficient or degenerate observations for ADF.",
        }


def _time_axis(series: pd.Series) -> tuple[dict[str, Any], pd.Series]:
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.dropna().sort_values()
    unique = valid.drop_duplicates()
    duplicate_count = int(valid.duplicated().sum())
    frequency = None
    if len(unique) >= 3:
        try:
            frequency = pd.infer_freq(unique)
        except ValueError:
            pass
    diffs = unique.diff().dropna()
    median_seconds = float(diffs.dt.total_seconds().median()) if len(diffs) else None
    gap_rows: list[dict[str, Any]] = []
    if median_seconds and median_seconds > 0:
        large = diffs[diffs.dt.total_seconds() > median_seconds * 1.5]
        for index, delta in large.nlargest(10).items():
            position = unique.index.get_loc(index)
            gap_rows.append(
                {
                    "start": unique.iloc[position - 1].isoformat(),
                    "end": unique.iloc[position].isoformat(),
                    "duration_seconds": float(delta.total_seconds()),
                    "expected_intervals": round(delta.total_seconds() / median_seconds, 2),
                }
            )
    regularity = None
    if len(diffs) and median_seconds:
        tolerance = max(median_seconds * 0.01, 1.0)
        regularity = float(np.mean(np.abs(diffs.dt.total_seconds() - median_seconds) <= tolerance))
    info = {
        "count": int(len(valid)),
        "invalid_or_missing": int(parsed.isna().sum()),
        "duplicate_timestamps": duplicate_count,
        "start": unique.min().isoformat() if len(unique) else None,
        "end": unique.max().isoformat() if len(unique) else None,
        "inferred_frequency": frequency,
        "median_interval_seconds": median_seconds,
        "regularity_ratio": regularity,
        "gaps": gap_rows,
    }
    return info, parsed


def _signal_diagnostics(
    frame: pd.DataFrame,
    time_column: str,
    value_column: str,
    frequency: Optional[str],
) -> Optional[dict[str, Any]]:
    data = frame[[time_column, value_column]].dropna().copy()
    if len(data) < 8 or data[value_column].nunique() < 2:
        return None
    data = data.groupby(time_column, as_index=False)[value_column].mean().sort_values(time_column)
    values = data[value_column].astype(float)
    index = np.arange(len(values), dtype=float)
    trend, trend_pvalue = stats.spearmanr(index, values)
    lag1 = float(values.autocorr(lag=1)) if len(values) > 2 else None
    period = _seasonal_period(frequency)
    seasonal_autocorrelation = None
    if period and len(values) >= period * 2:
        candidate = values.autocorr(lag=period)
        seasonal_autocorrelation = float(candidate) if np.isfinite(candidate) else None
    midpoint = len(values) // 2
    first_mean = float(values.iloc[:midpoint].mean())
    second_mean = float(values.iloc[midpoint:].mean())
    scale = float(values.std())
    mean_shift = (second_mean - first_mean) / scale if scale > 0 else 0.0
    return {
        "time_column": time_column,
        "value_column": value_column,
        "observations": int(len(values)),
        "trend_spearman": float(trend),
        "trend_pvalue": float(trend_pvalue),
        "lag1_autocorrelation": lag1,
        "seasonal_period": period,
        "seasonal_autocorrelation": seasonal_autocorrelation,
        "standardized_mean_shift": float(mean_shift),
        "stationarity": _adf(values),
    }


def _target_over_time(
    frame: pd.DataFrame, time_column: str, target: str, classification_threshold: int
) -> Optional[dict[str, Any]]:
    data = frame[[time_column, target]].dropna().sort_values(time_column)
    if len(data) < 20 or data[target].nunique() < 2:
        return None
    midpoint = len(data) // 2
    period = pd.Series(
        np.where(np.arange(len(data)) < midpoint, "first_half", "second_half"),
        index=data.index,
    )
    if (
        not pd.api.types.is_numeric_dtype(data[target])
        or data[target].nunique() <= classification_threshold
    ):
        table = pd.crosstab(period, data[target])
        pvalue = float(stats.chi2_contingency(table)[1]) if min(table.shape) >= 2 else None
        return {
            "time_column": time_column,
            "target": target,
            "target_type": "classification",
            "test": "chi_square_first_vs_second_half",
            "pvalue": pvalue,
            "temporal_shift": bool(pvalue is not None and pvalue < 0.05),
        }
    first = data[target].iloc[:midpoint].astype(float)
    second = data[target].iloc[midpoint:].astype(float)
    pvalue = float(stats.mannwhitneyu(first, second, alternative="two-sided").pvalue)
    pooled = float(data[target].std())
    effect = float((second.mean() - first.mean()) / pooled) if pooled > 0 else 0.0
    return {
        "time_column": time_column,
        "target": target,
        "target_type": "regression",
        "test": "mann_whitney_first_vs_second_half",
        "pvalue": pvalue,
        "standardized_shift": effect,
        "temporal_shift": bool(pvalue < 0.05 and abs(effect) >= 0.2),
    }


def time_series_diagnostics(
    df: pd.DataFrame,
    *,
    time_columns: list[str],
    target: Optional[str] = None,
    classification_threshold: int = 10,
    max_signals: int = 20,
) -> dict[str, Any]:
    """Diagnose temporal structure and recommend leakage-safe modeling actions."""
    axes: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    target_results: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    numeric = [
        str(column)
        for column in df.select_dtypes(include=np.number).columns
        if column not in time_columns
    ][:max_signals]
    for time_column in time_columns:
        if time_column not in df.columns:
            continue
        axis, parsed = _time_axis(df[time_column])
        axis["column"] = time_column
        axes.append(axis)
        working = df.copy()
        working[time_column] = parsed
        frequency = axis["inferred_frequency"]
        for value_column in numeric:
            signal = _signal_diagnostics(working, time_column, value_column, frequency)
            if signal:
                signals.append(signal)
        if target and target in df.columns:
            result = _target_over_time(
                working,
                time_column,
                target,
                classification_threshold,
            )
            if result:
                target_results.append(result)
        if axis["duplicate_timestamps"]:
            actions.append(
                {
                    "priority": "high",
                    "time_column": time_column,
                    "action": "define_duplicate_timestamp_aggregation_rule",
                    "evidence": f"{axis['duplicate_timestamps']} duplicate timestamps",
                }
            )
        if axis["gaps"]:
            actions.append(
                {
                    "priority": "medium",
                    "time_column": time_column,
                    "action": "reindex_expected_frequency_and_flag_gaps",
                    "evidence": f"{len(axis['gaps'])} relevant gaps detected",
                }
            )
        if axis["regularity_ratio"] is not None and axis["regularity_ratio"] < 0.8:
            actions.append(
                {
                    "priority": "medium",
                    "time_column": time_column,
                    "action": "validate_irregular_sampling_before_resampling",
                    "evidence": f"regularity ratio={axis['regularity_ratio']:.3f}",
                }
            )
    if axes:
        actions.extend(
            [
                {
                    "priority": "high",
                    "time_column": None,
                    "action": "use_rolling_origin_or_time_series_split_never_random_split",
                    "evidence": "temporal ordering detected",
                },
                {
                    "priority": "high",
                    "time_column": None,
                    "action": "shift_lag_and_rolling_features_before_training",
                    "evidence": "prevents future information leakage",
                },
                {
                    "priority": "medium",
                    "time_column": None,
                    "action": "fit_imputation_scaling_and_encoding_per_training_window",
                    "evidence": "prevents preprocessing leakage across time",
                },
            ]
        )
    for result in target_results:
        if result["temporal_shift"]:
            actions.append(
                {
                    "priority": "high",
                    "time_column": result["time_column"],
                    "action": "monitor_target_shift_and_use_recent_validation_window",
                    "evidence": f"{result['test']} p-value={result['pvalue']:.4g}",
                }
            )
    order = {"high": 0, "medium": 1, "low": 2}
    actions.sort(key=lambda row: order.get(str(row["priority"]), 9))
    return {
        "summary": {
            "time_columns": len(axes),
            "signals_analyzed": len(signals),
            "axes_with_gaps": sum(bool(axis["gaps"]) for axis in axes),
            "duplicate_timestamps": sum(axis["duplicate_timestamps"] for axis in axes),
            "non_stationary_signals": sum(
                signal["stationarity"].get("stationary") is False for signal in signals
            ),
            "target_temporal_shifts": sum(result["temporal_shift"] for result in target_results),
            "prioritized_actions": len(actions),
        },
        "time_axes": axes,
        "signals": signals,
        "target_temporal_analysis": target_results,
        "prioritized_actions": actions,
        "disclaimer": (
            "Temporal diagnostics are screening tools. Confirm frequency, exogenous availability, "
            "forecast horizon and backtesting design with domain knowledge."
        ),
    }
