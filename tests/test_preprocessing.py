import numpy as np
import pandas as pd

from smarteda.analysis.preprocessing import preprocessing_diagnostics


def test_preprocessing_detects_missing_outliers_normality_and_target_actions():
    rng = np.random.default_rng(42)
    size = 200
    target = np.array([0] * 160 + [1] * 40)
    feature = rng.normal(size=size)
    feature[target == 1] = np.nan
    outlier = rng.normal(size=size)
    outlier[:5] = 50
    skewed = rng.lognormal(size=size)
    result = preprocessing_diagnostics(
        pd.DataFrame({"feature": feature, "outlier": outlier, "skewed": skewed, "target": target}),
        target="target",
    )
    assert result["summary"]["columns_with_missing"] == 1
    assert result["summary"]["columns_with_outliers"] >= 1
    assert result["summary"]["non_normal_numeric_columns"] >= 1
    assert any(row.get("target_associated") for row in result["missing_data"])
    assert any("stratified_split" in row["action"] for row in result["target_actions"])


def test_preprocessing_never_mutates_input():
    frame = pd.DataFrame({"x": [1.0, 2.0, np.nan, 100.0], "target": [0, 0, 1, 1]})
    original = frame.copy(deep=True)
    preprocessing_diagnostics(frame, target="target")
    pd.testing.assert_frame_equal(frame, original)
