import numpy as np
import pandas as pd

from smarteda.analysis.time_series import time_series_diagnostics


def test_time_series_detects_structure_target_shift_and_safe_actions():
    dates = pd.date_range("2025-01-01", periods=120, freq="D").delete([40, 41])
    dates = dates.insert(10, dates[10])
    size = len(dates)
    target = np.array([0] * (size // 2) + [1] * (size - size // 2))
    frame = pd.DataFrame(
        {
            "date": dates,
            "value": np.arange(size) + np.sin(np.arange(size) * 2 * np.pi / 7),
            "target": target,
        }
    )
    result = time_series_diagnostics(frame, time_columns=["date"], target="target")
    assert result["summary"]["time_columns"] == 1
    assert result["summary"]["duplicate_timestamps"] == 1
    assert result["summary"]["axes_with_gaps"] == 1
    assert result["summary"]["target_temporal_shifts"] == 1
    actions = [row["action"] for row in result["prioritized_actions"]]
    assert "use_rolling_origin_or_time_series_split_never_random_split" in actions
    assert "shift_lag_and_rolling_features_before_training" in actions


def test_time_series_without_temporal_columns_returns_empty_diagnostics():
    result = time_series_diagnostics(pd.DataFrame({"value": [1, 2, 3]}), time_columns=[])
    assert result["summary"]["time_columns"] == 0
    assert result["prioritized_actions"] == []
