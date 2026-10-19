"""Layer manifest and the local (DuckDB) runner for silver and gold.

The manifest is shared with the Databricks notebooks so both environments
build exactly the same tables in the same order.
"""

from __future__ import annotations

import time
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from .sql_compat import render

if TYPE_CHECKING:
    import duckdb

__all__ = ["BRONZE_TABLES", "DEFAULT_PARAMS", "LAYERS", "build_layers", "load_sql", "render"]

BRONZE_TABLES = [
    "records",
    "hrv_instantaneous_bpm",
    "correlations",
    "workouts",
    "workout_events",
    "workout_statistics",
    "activity_summaries",
    "me",
    "ecg_readings",
    "ecg_samples",
    "route_points",
    "clinical_records",
]

# Ordered: a table may only depend on tables listed before it.
LAYERS: dict[str, list[str]] = {
    "silver": [
        "silver_records",
        "silver_sleep_segments",
        "silver_workout_statistics",
        "silver_workouts",
        "silver_workout_events",
        "silver_activity_summaries",
        "silver_hrv_beats",
        "silver_ecg_readings",
        "silver_ecg_samples",
        "silver_route_points",
        "silver_clinical_records",
        "silver_profile",
    ],
    "gold": [
        "gold_hourly_activity",
        "gold_daily_activity",
        "gold_hourly_heart_rate",
        "gold_daily_vitals",
        "gold_daily_sleep",
        "gold_workouts",
        "gold_daily_workouts",
        "gold_daily_summary",
        "gold_weekly_summary",
        "gold_ecg_summary",
    ],
}

DEFAULT_PARAMS = {"TIMEZONE": "Europe/Berlin"}


def load_sql(layer: str, table: str) -> str:
    return resources.files("health_lakehouse").joinpath(f"sql/{layer}/{table}.sql").read_text()


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    # Imported lazily: Databricks only needs the manifest, not DuckDB.
    import duckdb

    from .sql_compat import prepare_connection

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    prepare_connection(con)
    return con


def register_bronze(con: duckdb.DuckDBPyConnection, bronze_dir: str | Path) -> None:
    bronze_dir = Path(bronze_dir)
    for name in BRONZE_TABLES:
        path = bronze_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing bronze file {path}; run the bronze step first")
        con.execute(f"CREATE OR REPLACE VIEW bronze_{name} AS SELECT * FROM read_parquet('{path.as_posix()}')")


def build_layers(
    db_path: str | Path,
    bronze_dir: str | Path,
    layers: tuple[str, ...] = ("silver", "gold"),
    params: dict[str, str] | None = None,
    export_parquet_dir: str | Path | None = None,
) -> dict[str, int]:
    """Build silver/gold tables into a DuckDB file (and optionally Parquet)."""
    from .sql_compat import to_duckdb

    params = {**DEFAULT_PARAMS, **(params or {})}
    con = connect(db_path)
    register_bronze(con, bronze_dir)
    counts: dict[str, int] = {}
    for layer in layers:
        for table in LAYERS[layer]:
            t0 = time.time()
            sql = to_duckdb(render(load_sql(layer, table), params))
            con.execute(f"CREATE OR REPLACE TABLE {table} AS {sql}")
            counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<28} {counts[table]:>12,} rows  {time.time() - t0:6.1f}s", flush=True)
            if export_parquet_dir:
                out = Path(export_parquet_dir) / layer / f"{table}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                con.execute(f"COPY {table} TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()
    return counts
