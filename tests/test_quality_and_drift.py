import pandas as pd
import pytest

from smarteda import SmartEDA, detect_quality_issues, profile_train_test


def test_quality_diagnostics_detect_ids_constants_and_target_copy():
    df = pd.DataFrame(
        {
            "id_cliente": [1, 2, 3, 4],
            "constant": ["x", "x", "x", "x"],
            "target_copy": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )
    diagnostics = detect_quality_issues(df, target="target")

    assert diagnostics["summary"]["constant_columns"] == 1
    assert diagnostics["summary"]["possible_id_columns"] == 1
    assert any(item["check"] == "exact_target_copy" for item in diagnostics["possible_leakage"])


def test_train_test_profile_detects_distribution_and_schema_changes():
    train = pd.DataFrame({"x": range(100), "segment": ["A"] * 90 + ["B"] * 10})
    test = pd.DataFrame({"x": range(100, 200), "segment": ["C"] * 100, "new": 1})

    comparison = profile_train_test(train, test)
    rows = {row["column"]: row for row in comparison["features"]}

    assert comparison["summary"]["new_in_test"] == ["new"]
    assert rows["x"]["drift_level"] == "high"
    assert rows["segment"]["unseen_category_rate"] == 1.0


def test_smarteda_accepts_optional_polars():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    eda = SmartEDA(frame)
    assert isinstance(eda.df, pd.DataFrame)


def test_smarteda_accepts_optional_duckdb():
    duckdb = pytest.importorskip("duckdb")
    relation = duckdb.sql("select * from (values (1, 'a'), (2, 'b')) t(x, y)")

    eda = SmartEDA(relation)
    assert isinstance(eda.df, pd.DataFrame)
