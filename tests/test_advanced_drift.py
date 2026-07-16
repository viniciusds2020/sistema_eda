import numpy as np
import pandas as pd

from smarteda import (
    distribution_tests,
    monitor_windows,
    target_conditioned_drift,
)


def test_distribution_tests_apply_fdr_correction():
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, 500),
            "shifted": rng.normal(0, 1, 500),
            "category": ["A"] * 250 + ["B"] * 250,
        }
    )
    test = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, 500),
            "shifted": rng.normal(3, 1, 500),
            "category": ["C"] * 500,
        }
    )

    result = distribution_tests(train, test, correction="fdr_bh")
    rows = {row["column"]: row for row in result["features"]}

    assert rows["shifted"]["significant"]
    assert rows["category"]["significant"]
    assert all(row["adjusted_pvalue"] >= row["pvalue"] for row in result["features"])


def test_target_conditioned_drift_compares_each_class():
    train = pd.DataFrame(
        {
            "feature": list(range(100)) + list(range(100)),
            "target": [0] * 100 + [1] * 100,
        }
    )
    test = pd.DataFrame(
        {
            "feature": list(range(100, 200)) + list(range(100)),
            "target": [0] * 100 + [1] * 100,
        }
    )

    result = target_conditioned_drift(
        train,
        test,
        target="target",
        min_samples=20,
    )

    assert result["summary"]["segments_compared"] == 2
    assert any(
        row["target_segment"] == "0" and row["drift_level"] == "high"
        for row in result["features"]
    )


def test_longitudinal_monitor_preserves_window_order():
    reference = pd.DataFrame({"x": range(100), "group": ["A"] * 100})
    windows = {
        "2026-01": pd.DataFrame({"x": range(100), "group": ["A"] * 100}),
        "2026-02": pd.DataFrame({"x": range(100, 200), "group": ["B"] * 100}),
    }

    result = monitor_windows(reference, windows)
    assert [row["window"] for row in result["windows"]] == ["2026-01", "2026-02"]
    assert result["windows"][1]["high_drift_features"] >= 1
