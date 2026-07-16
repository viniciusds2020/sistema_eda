"""Adapters for bounded conversion from pandas, Polars, and DuckDB."""

from __future__ import annotations

from typing import Any

import pandas as pd


def to_pandas(
    frame: Any,
    *,
    copy: bool = True,
    max_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Convert a supported table to pandas with optional bounded materialization.

    For pandas and eager Polars frames, rows are sampled reproducibly. For
    Polars LazyFrame and DuckDB relations, limit is pushed down before collect
    or conversion, avoiding full source materialization.
    """
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive or None.")

    if isinstance(frame, pd.DataFrame):
        converted = frame
        if max_rows is not None and len(converted) > max_rows:
            converted = converted.sample(n=max_rows, random_state=random_state)
        return converted.copy() if copy else converted

    module = frame.__class__.__module__.split(".", 1)[0]

    if module == "polars":
        if frame.__class__.__name__ == "LazyFrame":
            bounded = frame.limit(max_rows) if max_rows is not None else frame
            frame = bounded.collect()
        elif max_rows is not None and frame.height > max_rows:
            frame = frame.sample(
                n=max_rows,
                seed=random_state,
                shuffle=True,
            )
        converted = frame.to_pandas()
        return converted.copy() if copy else converted

    if module in {"duckdb", "_duckdb"} and hasattr(frame, "df"):
        bounded = frame.limit(max_rows) if max_rows is not None else frame
        converted = bounded.df()
        if not isinstance(converted, pd.DataFrame):
            raise TypeError("DuckDB relation did not return a pandas DataFrame.")
        return converted.copy() if copy else converted

    raise TypeError(
        "Unsupported data object. Expected pandas.DataFrame, "
        "polars.DataFrame/LazyFrame, or duckdb.DuckDBPyRelation."
    )
