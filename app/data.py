"""Data access for the Streamlit app, with two interchangeable backends.

HEALTH_BACKEND=local (default)
    In-memory DuckDB over the Parquet files in HEALTH_DATA_DIR
    (default: <repo>/data). Every file is exposed as a view named after its
    stem (`gold_workouts`, `silver_route_points`, `ml_daily_anomalies`, ...).

HEALTH_BACKEND=databricks
    databricks-sql-connector against the SQL warehouse DATABRICKS_WAREHOUSE_ID,
    authenticated as the Databricks App's service principal. Tables live in
    HEALTH_CATALOG.HEALTH_SCHEMA; ML outputs are the `ml_<stem>` tables there.

Every query string is written once, in the SQL subset that DuckDB and
Databricks SQL share, against names produced by `table()`.
"""

from __future__ import annotations

import datetime as dt
import decimal
import os
import re
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DATE_COLUMNS = ("local_date", "night_date", "week_start")


# --------------------------------------------------------------------------- config


def backend() -> str:
    value = os.environ.get("HEALTH_BACKEND", "local").strip().lower()
    if value not in ("local", "databricks"):
        raise ValueError(f"HEALTH_BACKEND must be 'local' or 'databricks', got {value!r}")
    return value


def data_dir() -> Path:
    default = Path(__file__).resolve().parent.parent / "data"
    return Path(os.environ.get("HEALTH_DATA_DIR", default)).expanduser().resolve()


def _catalog_schema() -> tuple[str, str]:
    catalog = os.environ.get("HEALTH_CATALOG", "workspace")
    schema = os.environ.get("HEALTH_SCHEMA", "health")
    for part in (catalog, schema):
        if not _IDENT.match(part):
            raise ValueError(f"invalid catalog/schema identifier: {part!r}")
    return catalog, schema


def schema_name() -> str:
    """`catalog.schema` holding the tables in Databricks."""
    return ".".join(_catalog_schema())


def table(name: str) -> str:
    """Resolve a logical table name to what the active backend can query."""
    if not _IDENT.match(name):
        raise ValueError(f"invalid table name: {name!r}")
    if backend() == "databricks":
        catalog, schema = _catalog_schema()
        return f"{catalog}.{schema}.{name}"
    return name


def ml_table(stem: str) -> str:
    """ML outputs: `data/ml/<stem>.parquet` locally, `ml_<stem>` in Databricks."""
    return table(f"ml_{stem}")


# --------------------------------------------------------------------------- literals


def date_lit(value: dt.date) -> str:
    if not isinstance(value, dt.date):
        raise TypeError(f"expected a date, got {type(value).__name__}")
    if isinstance(value, dt.datetime):
        value = value.date()
    return f"DATE '{value.isoformat()}'"


def str_lit(value: str) -> str:
    """Quoted string literal. Databricks treats backslash as an escape, DuckDB does not."""
    value = str(value)
    if backend() == "databricks":
        value = value.replace("\\", "\\\\")
    return "'" + value.replace("'", "''") + "'"


# --------------------------------------------------------------------------- engines

_local_lock = threading.Lock()


@st.cache_resource(show_spinner=False)
def _duckdb(root: str):
    """One in-memory DuckDB per data directory (views point at its files)."""
    import duckdb

    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone = 'UTC'")
    return con


def _register_local_views(con, root: Path) -> None:
    existing = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    for layer in ("silver", "gold", "ml"):
        for path in sorted((root / layer).glob("*.parquet")):
            name = f"ml_{path.stem}" if layer == "ml" else path.stem
            if name in existing or not _IDENT.match(name):
                continue
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({str_lit(path.as_posix())})")


def _run_local(sql: str) -> pd.DataFrame:
    root = data_dir()
    con = _duckdb(str(root))
    with _local_lock:
        _register_local_views(con, root)
        return con.cursor().execute(sql).df()


def _run_databricks(sql: str) -> pd.DataFrame:
    from databricks.sdk.core import Config

    from databricks import sql as dbsql

    cfg = Config()  # picks up the app service principal (or local CLI profile)
    with (
        dbsql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}",
            credentials_provider=lambda: cfg.authenticate,
        ) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(sql)
        return cur.fetchall_arrow().to_pandas()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Make both backends return the same dtypes (dates as datetime64, no Decimals)."""
    for col in df.columns:
        if df[col].dtype != object:
            continue
        sample = df[col].dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        if isinstance(first, decimal.Decimal):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        elif isinstance(first, (dt.date, dt.datetime)) or col in _DATE_COLUMNS:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in _DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    # *_local columns are TIMESTAMP_NTZ (naive wall clock) on both backends. A
    # tz-aware TIMESTAMP (e.g. *_utc) is made naive UTC so comparisons never mix kinds.
    for col in df.columns:
        if isinstance(df[col].dtype, pd.DatetimeTZDtype):
            df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def source_key() -> str:
    """Identifies where data comes from; part of every cache key."""
    if backend() == "databricks":
        return f"databricks:{os.environ.get('DATABRICKS_WAREHOUSE_ID', '')}:{schema_name()}"
    return f"local:{data_dir()}"


def _execute(sql: str) -> pd.DataFrame:
    df = _run_databricks(sql) if backend() == "databricks" else _run_local(sql)
    return _normalize(df)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached(sql: str, source: str) -> pd.DataFrame:
    return _execute(sql)


def run(sql: str) -> pd.DataFrame:
    return _cached(sql, source_key())


# --------------------------------------------------------------------------- queries
# Each function builds one SQL string (see `*_sql`) and runs it, so tests can
# parse the exact strings the app sends to Databricks.


def date_bounds_sql() -> str:
    return f"SELECT MIN(local_date) AS min_date, MAX(local_date) AS max_date FROM {table('gold_daily_summary')}"


def date_bounds() -> tuple[dt.date, dt.date]:
    row = run(date_bounds_sql()).iloc[0]
    return pd.Timestamp(row["min_date"]).date(), pd.Timestamp(row["max_date"]).date()


def daily_summary_sql(start: dt.date, end: dt.date) -> str:
    return (
        f"SELECT * FROM {table('gold_daily_summary')} "
        f"WHERE local_date BETWEEN {date_lit(start)} AND {date_lit(end)} ORDER BY local_date"
    )


def daily_summary(start: dt.date, end: dt.date) -> pd.DataFrame:
    return run(daily_summary_sql(start, end))


def coverage_sql() -> str:
    metrics = [
        "steps",
        "active_kcal",
        "exercise_min",
        "hr_mean",
        "resting_hr",
        "hrv_sdnn_ms",
        "vo2max",
        "spo2_pct",
        "body_mass_kg",
        "walking_speed_kmh",
        "sleep_in_bed_min",
        "sleep_asleep_min",
        "move_goal_kcal",
    ]
    cols = ", ".join(f"COUNT({m}) AS {m}" for m in metrics)
    return f"SELECT year, COUNT(*) AS days, {cols} FROM {table('gold_daily_summary')} GROUP BY year ORDER BY year"


def coverage() -> pd.DataFrame:
    return run(coverage_sql())


def weekly_summary_sql(start: dt.date, end: dt.date) -> str:
    return (
        f"SELECT * FROM {table('gold_weekly_summary')} "
        f"WHERE week_start BETWEEN {date_lit(start)} AND {date_lit(end)} ORDER BY week_start"
    )


def weekly_summary(start: dt.date, end: dt.date) -> pd.DataFrame:
    return run(weekly_summary_sql(start, end))


HOURLY_METRICS = {
    "StepCount": "Steps",
    "ActiveEnergyBurned": "Active energy (kcal)",
    "AppleExerciseTime": "Exercise (min)",
    "AppleStandTime": "Stand (min)",
    "DistanceWalkingRunning": "Walking + running (km)",
    "FlightsClimbed": "Flights climbed",
}


def hourly_activity_profile_sql(metric: str, start: dt.date, end: dt.date) -> str:
    """Average value per (weekday, hour); hours without records count as zero."""
    if metric not in HOURLY_METRICS:
        raise ValueError(f"unknown hourly metric {metric!r}")
    return f"""
WITH h AS (
  SELECT local_date, local_hour, value
  FROM {table("gold_hourly_activity")}
  WHERE metric = {str_lit(metric)}
    AND local_date BETWEEN {date_lit(start)} AND {date_lit(end)}
),
d AS (
  SELECT local_date, day_of_week
  FROM {table("gold_daily_summary")}
  WHERE local_date IN (SELECT DISTINCT local_date FROM h)
),
n AS (
  SELECT day_of_week, COUNT(*) AS n_days FROM d GROUP BY day_of_week
)
SELECT d.day_of_week, h.local_hour, SUM(h.value) / MAX(n.n_days) AS avg_value, MAX(n.n_days) AS n_days
FROM h
JOIN d ON h.local_date = d.local_date
JOIN n ON d.day_of_week = n.day_of_week
GROUP BY d.day_of_week, h.local_hour
ORDER BY d.day_of_week, h.local_hour
"""


def hourly_activity_profile(metric: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    return run(hourly_activity_profile_sql(metric, start, end))


def hourly_heart_rate_profile_sql(start: dt.date, end: dt.date) -> str:
    return f"""
SELECT d.day_of_week, h.local_hour,
       SUM(h.hr_mean * h.n_samples) / SUM(h.n_samples) AS hr_mean,
       MIN(h.hr_min) AS hr_min,
       MAX(h.hr_max) AS hr_max,
       SUM(h.n_samples) AS n_samples
FROM {table("gold_hourly_heart_rate")} h
JOIN {table("gold_daily_summary")} d ON h.local_date = d.local_date
WHERE h.local_date BETWEEN {date_lit(start)} AND {date_lit(end)}
GROUP BY d.day_of_week, h.local_hour
ORDER BY d.day_of_week, h.local_hour
"""


def hourly_heart_rate_profile(start: dt.date, end: dt.date) -> pd.DataFrame:
    return run(hourly_heart_rate_profile_sql(start, end))


def workouts_sql(start: dt.date, end: dt.date) -> str:
    return (
        f"SELECT * FROM {table('gold_workouts')} "
        f"WHERE local_date BETWEEN {date_lit(start)} AND {date_lit(end)} ORDER BY start_local DESC"
    )


def workouts(start: dt.date, end: dt.date) -> pd.DataFrame:
    return run(workouts_sql(start, end))


def workout_hr_reference_sql() -> str:
    """All-time high workout heart rate, the HRmax estimate for zone proxies."""
    return (
        f"SELECT percentile_cont(0.98) WITHIN GROUP (ORDER BY hr_max) AS hr_max_ref, COUNT(hr_max) AS n "
        f"FROM {table('gold_workouts')} WHERE hr_max BETWEEN 100 AND 230"
    )


def workout_hr_reference() -> float | None:
    df = run(workout_hr_reference_sql())
    value = df["hr_max_ref"].iloc[0] if not df.empty else None
    return None if value is None or pd.isna(value) else float(value)


def route_points_sql(route_file: str) -> str:
    return (
        f"SELECT point_index, lat, lon, elevation_m, point_local, speed_mps "
        f"FROM {table('silver_route_points')} "
        f"WHERE route_file = {str_lit(route_file)} "
        f"AND (h_accuracy_m IS NULL OR h_accuracy_m < 50) ORDER BY point_index"
    )


def route_points(route_file: str) -> pd.DataFrame:
    return run(route_points_sql(route_file))


def sleep_sql(start: dt.date, end: dt.date) -> str:
    return (
        f"SELECT * FROM {table('gold_daily_sleep')} "
        f"WHERE night_date BETWEEN {date_lit(start)} AND {date_lit(end)} ORDER BY night_date"
    )


def sleep(start: dt.date, end: dt.date) -> pd.DataFrame:
    return run(sleep_sql(start, end))


def ecg_list_sql() -> str:
    return f"SELECT * FROM {table('gold_ecg_summary')} ORDER BY recorded_local DESC"


def ecg_list() -> pd.DataFrame:
    return run(ecg_list_sql())


def ecg_samples_sql(ecg_id: str) -> str:
    return (
        f"SELECT sample_index, value_uv FROM {table('silver_ecg_samples')} "
        f"WHERE ecg_id = {str_lit(ecg_id)} ORDER BY sample_index"
    )


def ecg_samples(ecg_id: str) -> pd.DataFrame:
    return run(ecg_samples_sql(ecg_id))


# --------------------------------------------------------------------------- ML outputs


def list_ml_tables_sql() -> str:
    catalog, schema = _catalog_schema()
    return f"SHOW TABLES IN {catalog}.{schema} LIKE 'ml_*'"


def list_ml_tables() -> list[str]:
    """Stems of the available ML output tables (`daily_anomalies`, ...)."""
    return _list_ml_tables(source_key())


@st.cache_data(ttl=60, show_spinner=False)
def _list_ml_tables(source: str) -> list[str]:
    if backend() == "databricks":
        df = _run_databricks(list_ml_tables_sql())
        names = df["tableName"] if "tableName" in df.columns else df.iloc[:, 1]
        return sorted(n[3:] for n in names if str(n).startswith("ml_"))
    folder = data_dir() / "ml"
    return sorted(p.stem for p in folder.glob("*.parquet") if _IDENT.match(p.stem))


def ml_frame_sql(stem: str, limit: int | None = None) -> str:
    sql = f"SELECT * FROM {ml_table(stem)}"
    return sql + (f" LIMIT {int(limit)}" if limit else "")


def ml_frame(stem: str, limit: int | None = None) -> pd.DataFrame:
    """ML outputs change when models are retrained, so they are cached briefly."""
    return _ml_cached(ml_frame_sql(stem, limit), source_key())


@st.cache_data(ttl=60, show_spinner=False)
def _ml_cached(sql: str, source: str) -> pd.DataFrame:
    return _execute(sql)
