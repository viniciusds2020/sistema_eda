import pandas as pd
import pytest

from smarteda import Config, SmartEDA, to_pandas


def test_pandas_materialization_limit_is_respected():
    frame = pd.DataFrame({"x": range(1000)})
    bounded = to_pandas(frame, max_rows=100, random_state=42)
    assert len(bounded) == 100


def test_polars_lazyframe_pushes_limit_before_collect():
    pl = pytest.importorskip("polars")
    lazy = pl.DataFrame({"x": range(1000)}).lazy()
    bounded = to_pandas(lazy, max_rows=75)
    assert len(bounded) == 75


def test_duckdb_relation_pushes_limit_before_conversion():
    duckdb = pytest.importorskip("duckdb")
    relation = duckdb.sql("select range as x from range(1000)")
    bounded = to_pandas(relation, max_rows=60)
    assert len(bounded) == 60


def test_smarteda_uses_sample_size_as_materialization_limit():
    frame = pd.DataFrame({"x": range(1000)})
    eda = SmartEDA(frame, config=Config(sample_size=80, include_plots=False))
    assert len(eda.df) == 80
