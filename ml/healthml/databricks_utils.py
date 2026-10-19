"""Helpers for the Databricks notebooks in databricks/notebooks/ml (no Spark import at module level)."""

from __future__ import annotations

import pandas as pd


def write_delta(spark, df: pd.DataFrame, catalog: str, schema: str, name: str) -> str:
    """Overwrite ``{catalog}.{schema}.{name}`` with the pandas frame (schema overwrite allowed)."""
    full = f"{catalog}.{schema}.{name}"
    clean = df.copy()
    for c in clean.columns:
        if clean[c].dtype == object and clean[c].map(lambda v: isinstance(v, (list, dict))).any():
            clean[c] = clean[c].astype(str)
    (spark.createDataFrame(clean).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(full))
    return full


def write_tables(spark, tables: dict[str, pd.DataFrame], catalog: str, schema: str) -> list[str]:
    """Write every output table as ``ml_<name>``."""
    written = []
    for name, df in tables.items():
        written.append(write_delta(spark, df, catalog, schema, f"ml_{name}"))
        print(f"wrote {written[-1]} ({len(df)} rows)")
    return written
