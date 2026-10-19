"""Run the silver/gold SQL manifest on Spark (Databricks notebooks and local tests).

Same files, same order as the DuckDB runner in `pipeline.py`, but the SQL is
executed as-is (no transpiling). Only needs PySpark at call time, so the
module can be imported on Databricks with just `src/` on `sys.path`.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from .pipeline import DEFAULT_PARAMS, LAYERS, load_sql, render

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

# Documented on every table that has the column (Unity Catalog column comments).
COLUMN_COMMENTS = {
    "record_id": "Stable SHA-1 of the HealthKit sample (type, source, dates, value)",
    "local_date": "Calendar date in the configured IANA timezone",
    "local_hour": "Hour of day (0-23) in the configured IANA timezone",
    "start_utc": "Sample start as an instant (UTC)",
    "end_utc": "Sample end as an instant (UTC)",
    "start_local": "Start as wall-clock time in the configured timezone (no zone attached)",
    "end_local": "End as wall-clock time in the configured timezone (no zone attached)",
    "metric": "HealthKit type without the HK...TypeIdentifier prefix",
    "value_num": "Numeric value; percentages scaled to 0-100",
    "night_date": "Date of the evening on which the night started",
    "week_start": "Monday of the ISO week",
    "steps": "Steps, deduplicated across iPhone/Watch per hour",
    "resting_hr": "Average resting heart rate (bpm)",
    "hrv_sdnn_ms": "Average heart rate variability SDNN (ms)",
    "workout_id": "Stable id of the workout",
}


def table_comment(sql: str) -> str:
    """The leading `--` comment block of a SQL file, joined into one line."""
    lines = []
    for line in sql.lstrip().splitlines():
        if not line.startswith("--"):
            break
        lines.append(line[2:].strip())
    return " ".join(lines)


def _quote(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def build_layer_spark(
    spark: SparkSession,
    layer: str,
    params: dict[str, str] | None = None,
    *,
    mode: str = "table",
    tables: list[str] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Create every table of `layer` in the current catalog/schema.

    mode="table": `CREATE OR REPLACE TABLE ... USING DELTA` with table and
    column comments (Databricks, or local Spark with delta-spark).
    mode="view": cached temp views, for quick local validation.
    """
    if mode not in ("table", "view"):
        raise ValueError(f"unknown mode {mode!r}")
    params = {**DEFAULT_PARAMS, **(params or {})}
    counts: dict[str, int] = {}
    for table in tables or LAYERS[layer]:
        t0 = time.time()
        raw = load_sql(layer, table)
        sql = render(raw, params)
        if re.search(r"\{\{\w+\}\}", sql):
            raise ValueError(f"{table}: unrendered placeholder in SQL")
        comment = table_comment(render(raw, params))
        if mode == "table":
            spark.sql(f"CREATE OR REPLACE TABLE {table} USING DELTA COMMENT {_quote(comment)} AS\n{sql}")
            columns = set(spark.table(table).columns)
            for col, text in COLUMN_COMMENTS.items():
                if col in columns:
                    spark.sql(f"ALTER TABLE {table} ALTER COLUMN {col} COMMENT {_quote(text)}")
        else:
            spark.sql(f"CREATE OR REPLACE TEMP VIEW {table} AS\n{sql}")
            spark.sql(f"CACHE TABLE {table}")
        counts[table] = spark.table(table).count()
        log(f"  {table:<28} {counts[table]:>12,} rows  {time.time() - t0:6.1f}s")
    return counts
