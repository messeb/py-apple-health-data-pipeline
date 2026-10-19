"""Where the lakehouse tables come from: local Parquet files or Unity Catalog tables via Spark.

Experiments only call ``source.load(name)`` and get a pandas DataFrame back, so the
same code runs locally and on Databricks.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import DATA_DIR


class ParquetSource:
    """Reads ``<data_dir>/gold/<name>.parquet`` or ``<data_dir>/silver/<name>.parquet`` (never the DuckDB file).

    ``data_dir`` defaults to ``HEALTH_DATA_DIR`` (read at construction time), else ``data/``.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or os.environ.get("HEALTH_DATA_DIR") or DATA_DIR).expanduser()
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        key = f"{name}:{columns}"
        if key not in self._cache:
            layer = name.split("_", 1)[0]
            path = self.data_dir / layer / f"{name}.parquet"
            self._cache[key] = pd.read_parquet(path, columns=columns)
        return self._cache[key].copy()


class SparkSource:
    """Reads ``{catalog}.{schema}.{name}`` with Spark and converts to pandas (tables are small)."""

    def __init__(self, spark, catalog: str = "workspace", schema: str = "health"):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self._cache: dict[str, pd.DataFrame] = {}

    def table_name(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    def load(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        key = f"{name}:{columns}"
        if key not in self._cache:
            df = self.spark.table(self.table_name(name))
            if columns:
                df = df.select(*columns)
            self._cache[key] = df.toPandas()
        return self._cache[key].copy()
