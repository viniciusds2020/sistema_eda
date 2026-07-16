"""Adapters for pandas, Polars, and DuckDB tabular objects."""

from __future__ import annotations

from typing import Any

import pandas as pd


def to_pandas(frame: Any, *, copy: bool = True) -> pd.DataFrame:
    """Convert a supported tabular object to a pandas DataFrame.

    Supported inputs:
    - pandas.DataFrame
    - polars.DataFrame and polars.LazyFrame
    - duckdb.DuckDBPyRelation
    """
    if isinstance(frame, pd.DataFrame):
        return frame.copy() if copy else frame

    module = frame.__class__.__module__.split(".", 1)[0]

    if module == "polars":
        if frame.__class__.__name__ == "LazyFrame":
            frame = frame.collect()
        converted = frame.to_pandas()
        return converted.copy() if copy else converted

    if module == "duckdb" and hasattr(frame, "df"):
        converted = frame.df()
        if not isinstance(converted, pd.DataFrame):
            raise TypeError("DuckDB relation did not return a pandas DataFrame.")
        return converted.copy() if copy else converted

    raise TypeError(
        "Unsupported data object. Expected pandas.DataFrame, "
        "polars.DataFrame/LazyFrame, or duckdb.DuckDBPyRelation."
    )
