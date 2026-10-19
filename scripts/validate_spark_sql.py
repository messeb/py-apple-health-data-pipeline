"""Check that the silver/gold SQL runs on Spark and matches the DuckDB results.

1. Cut a sample of the bronze Parquet files (records from date windows around
   the DST switches in the data plus its last fortnight, and the small tables)
   into `<out>/bronze/`.
2. Build silver + gold from the sample with DuckDB (sqlglot transpile path).
3. Build silver + gold from the same sample on a local SparkSession with the
   SQL files as-is (session timezone UTC, ANSI mode on, Delta tables when
   delta-spark is installed), exactly like the Databricks notebooks.
4. Compare every table row by row, and check local times against Python's
   zoneinfo as an independent reference.

Spark 4 needs Java 17 or 21. If `JAVA_HOME` is not one of those, a JDK under
`.cache/jdk/` is used.

    .venv/bin/python scripts/validate_spark_sql.py [--bronze-dir data/bronze] [--out data/_spark_test]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from health_lakehouse.pipeline import BRONZE_TABLES, DEFAULT_PARAMS, LAYERS, build_layers

MAX_DST_WINDOWS = 4

# Natural keys for row-by-row comparison; other tables are compared sorted by all columns.
KEYS = {
    "silver_records": ["record_id"],
    "silver_sleep_segments": ["record_id"],
    "silver_workouts": ["workout_id"],
    "silver_activity_summaries": ["local_date"],
    "silver_ecg_readings": ["ecg_id"],
    "silver_ecg_samples": ["ecg_id", "sample_index"],
    "silver_route_points": ["route_file", "point_index"],
    "gold_hourly_activity": ["local_date", "local_hour", "metric"],
    "gold_daily_activity": ["local_date"],
    "gold_hourly_heart_rate": ["local_date", "local_hour"],
    "gold_daily_vitals": ["local_date"],
    "gold_daily_sleep": ["night_date"],
    "gold_workouts": ["workout_id"],
    "gold_daily_workouts": ["local_date"],
    "gold_daily_summary": ["local_date"],
    "gold_weekly_summary": ["week_start"],
    "gold_ecg_summary": ["ecg_id"],
}


# --------------------------------------------------------------------------- java


def _java_major(java_home: Path) -> int | None:
    java = java_home / "bin" / "java"
    if not java.exists():
        return None
    out = subprocess.run([str(java), "-version"], capture_output=True, text=True).stderr
    m = re.search(r'version "(\d+)', out)
    return int(m.group(1)) if m else None


def find_java_home() -> Path | None:
    """A Java 17/21 home: $JAVA_HOME if suitable, else one under .cache/jdk/."""
    candidates = []
    if os.environ.get("JAVA_HOME"):
        candidates.append(Path(os.environ["JAVA_HOME"]))
    for jdk in sorted((REPO / ".cache" / "jdk").glob("jdk-*"), reverse=True):
        candidates += [jdk / "Contents" / "Home", jdk]  # macOS / Linux layout
    for home in candidates:
        if _java_major(home) in (17, 21):
            return home
    return None


# ------------------------------------------------------------------------- sample


def dst_windows(first: date, last: date, tz: str, max_windows: int = MAX_DST_WINDOWS) -> list[tuple[str, str]]:
    """Date windows around the latest DST switches of `tz` between `first` and `last`, plus the last 14 days."""
    zone = ZoneInfo(tz)
    switches, day = [], first
    while day < last:
        nxt = day + timedelta(days=1)
        if (
            datetime(day.year, day.month, day.day, 12, tzinfo=zone).utcoffset()
            != datetime(nxt.year, nxt.month, nxt.day, 12, tzinfo=zone).utcoffset()
        ):
            switches.append(nxt)
        day = nxt
    windows = [(str(d - timedelta(days=5)), str(d + timedelta(days=5))) for d in switches[-max_windows:]]
    return [*windows, (str(last - timedelta(days=13)), str(last))]


def default_windows(bronze_dir: Path, tz: str) -> list[tuple[str, str]]:
    import duckdb

    path = (bronze_dir / "records.parquet").as_posix()
    lo, hi = (
        duckdb.connect()
        .execute(
            f"SELECT min(substring(start_date, 1, 10)), max(substring(start_date, 1, 10)) FROM read_parquet('{path}')"
        )
        .fetchone()
    )
    return dst_windows(date.fromisoformat(lo), date.fromisoformat(hi), tz)


def make_sample(
    bronze_dir: Path, out_dir: Path, windows=None, max_records: int = 200_000, tz: str = DEFAULT_PARAMS["TIMEZONE"]
) -> dict[str, int]:
    """Write a consistent subset of the bronze tables to `out_dir`."""
    import duckdb

    out_dir.mkdir(parents=True, exist_ok=True)
    windows = windows or default_windows(bronze_dir, tz)
    con = duckdb.connect()
    src = {t: (bronze_dir / f"{t}.parquet").as_posix() for t in BRONZE_TABLES}
    in_windows = " OR ".join(f"substring({{col}}, 1, 10) BETWEEN '{a}' AND '{b}'" for a, b in windows)
    queries = {t: f"SELECT * FROM read_parquet('{src[t]}')" for t in BRONZE_TABLES}
    queries["records"] = (
        f"SELECT * FROM read_parquet('{src['records']}') WHERE {in_windows.format(col='start_date')} LIMIT {max_records}"
    )
    queries["hrv_instantaneous_bpm"] = (
        f"SELECT * FROM read_parquet('{src['hrv_instantaneous_bpm']}') "
        f"WHERE record_id IN (SELECT record_id FROM read_parquet('{(out_dir / 'records.parquet').as_posix()}'))"
    )
    queries["route_points"] = (
        f"SELECT * FROM read_parquet('{src['route_points']}') WHERE route_file IN ("
        f"SELECT route_file FROM read_parquet('{src['workouts']}') WHERE {in_windows.format(col='start_date')})"
    )
    counts = {}
    for t in ["records"] + [t for t in BRONZE_TABLES if t != "records"]:
        path = out_dir / f"{t}.parquet"
        con.execute(f"COPY ({queries[t]}) TO '{path.as_posix()}' (FORMAT PARQUET)")
        counts[t] = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]
    return counts


# -------------------------------------------------------------------------- spark


def spark_session(warehouse: Path, use_delta: bool | None = None):
    """Local SparkSession configured like the Databricks job (UTC, ANSI on)."""
    os.environ["TZ"] = "UTC"
    time.tzset()
    java_home = find_java_home()
    if java_home is None:
        raise RuntimeError("no Java 17/21 found (set JAVA_HOME or unpack a JDK into .cache/jdk/)")
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    from pyspark.sql import SparkSession

    if use_delta is None:
        try:
            import delta  # noqa: F401

            use_delta = True
        except ImportError:
            use_delta = False
    builder = (
        SparkSession.builder.master("local[4]")
        .appName("validate_spark_sql")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.driver.memory", "4g")
        .config("spark.ui.enabled", "false")
    )
    if use_delta:
        from delta import configure_spark_with_delta_pip

        builder = configure_spark_with_delta_pip(
            builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension").config(
                "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
            )
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark, use_delta


def build_spark(spark, sample_dir: Path, mode: str, params: dict[str, str]) -> dict[str, int]:
    from health_lakehouse.spark_runner import build_layer_spark

    for t in BRONZE_TABLES:
        spark.read.parquet(str(sample_dir / f"{t}.parquet")).createOrReplaceTempView(f"bronze_{t}")
    counts = {}
    for layer in ("silver", "gold"):
        counts |= build_layer_spark(spark, layer, params, mode=mode)
    return counts


# ---------------------------------------------------------------------- compare


@dataclass
class TableDiff:
    table: str
    duck_rows: int
    spark_rows: int
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = s.astype("datetime64[us]")
            continue
        non_null = s.dropna()
        if len(non_null) and isinstance(non_null.iloc[0], Decimal):
            s = s.map(lambda v: None if v is None else float(v)).astype("float64")
        elif len(non_null) and type(non_null.iloc[0]).__name__ == "date":
            s = pd.to_datetime(s).astype("datetime64[us]")
        elif pd.api.types.is_bool_dtype(s) or (len(non_null) and isinstance(non_null.iloc[0], (bool, np.bool_))):
            s = s.astype("boolean")
        elif pd.api.types.is_numeric_dtype(s):
            s = s.astype("float64")
        out[col] = s
    return pd.DataFrame(out)


def _columns_equal(a: pd.Series, b: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
        av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
        return pd.Series(np.isclose(av, bv, rtol=1e-9, atol=1e-9) | (np.isnan(av) & np.isnan(bv)), index=a.index)
    both_null = a.isna() & b.isna()
    return both_null | (a.astype(object) == b.astype(object)).fillna(False).astype(bool)


def compare_table(table: str, duck: pd.DataFrame, spark: pd.DataFrame) -> TableDiff:
    diff = TableDiff(table, len(duck), len(spark))
    if set(duck.columns) != set(spark.columns):
        diff.problems.append(
            f"columns differ: duckdb-only {set(duck.columns) - set(spark.columns)}, "
            f"spark-only {set(spark.columns) - set(duck.columns)}"
        )
    cols = [c for c in duck.columns if c in spark.columns]
    duck, spark = _normalise(duck[cols]), _normalise(spark[cols])
    if len(duck) != len(spark):
        diff.problems.append(f"row count {len(duck)} (duckdb) vs {len(spark)} (spark)")
    keys = KEYS.get(table) or cols
    sort_keys = [k for k in keys if k in cols]

    def _sorted(df):
        tmp = df.assign(**{f"__k{i}": df[k].astype(str) for i, k in enumerate(sort_keys)})
        return (
            tmp.sort_values([f"__k{i}" for i in range(len(sort_keys))])
            .drop(columns=[f"__k{i}" for i in range(len(sort_keys))])
            .reset_index(drop=True)
        )

    if table in KEYS:
        dup = duck.duplicated(sort_keys).sum() + spark.duplicated(sort_keys).sum()
        if dup:
            diff.problems.append(f"{dup} duplicate keys {sort_keys}")
        duck, spark = duck.copy(), spark.copy()
        m = duck.astype({k: str for k in sort_keys}).merge(
            spark.astype({k: str for k in sort_keys}), on=sort_keys, how="outer", suffixes=("_d", "_s"), indicator=True
        )
        only = m["_merge"].value_counts()
        if only.get("left_only", 0) or only.get("right_only", 0):
            diff.problems.append(
                f"keys only in duckdb: {only.get('left_only', 0)}, only in spark: {only.get('right_only', 0)}"
                f" e.g. {m.loc[m['_merge'] != 'both', sort_keys].head(3).to_dict('records')}"
            )
        m = m[m["_merge"] == "both"].reset_index(drop=True)
        pairs = [(m[f"{c}_d"], m[f"{c}_s"], c) for c in cols if c not in sort_keys]
        key_frame = m[sort_keys]
    else:
        a, b = _sorted(duck), _sorted(spark)
        n = min(len(a), len(b))
        pairs = [(a[c].iloc[:n], b[c].iloc[:n], c) for c in cols]
        key_frame = a.iloc[:n]
    for a_col, b_col, c in pairs:
        eq = _columns_equal(a_col.reset_index(drop=True), b_col.reset_index(drop=True))
        bad = int((~eq).sum())
        if bad:
            i = int(np.flatnonzero(~eq.to_numpy())[0])
            where = key_frame.iloc[i].to_dict() if table in KEYS else f"row {i}"
            diff.problems.append(
                f"{c}: {bad} rows differ, e.g. at {where}: duckdb={a_col.iloc[i]!r} spark={b_col.iloc[i]!r}"
            )
    return diff


def expected_local(date_str: str, tz: str) -> datetime:
    """Independent reference: Apple's 'YYYY-MM-DD HH:MM:SS +HHMM' -> local wall clock."""
    aware = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")
    return aware.astimezone(ZoneInfo(tz)).replace(tzinfo=None)


def check_against_zoneinfo(spark, tz: str, n: int = 2000) -> list[str]:
    rows = spark.sql(f"""
        SELECT b.start_date, r.start_local, r.local_date, r.local_hour
        FROM silver_records r JOIN bronze_records b USING (record_id)
        ORDER BY r.record_id LIMIT {n}""").collect()
    problems = []
    for row in rows:
        exp = expected_local(row.start_date, tz)
        if (row.start_local, row.local_date, row.local_hour) != (exp, exp.date(), exp.hour):
            problems.append(
                f"{row.start_date} -> spark {row.start_local} / {row.local_date} {row.local_hour}h, expected {exp}"
            )
    return problems


# ------------------------------------------------------------------------- main


@dataclass
class Report:
    sample_counts: dict[str, int]
    duck_counts: dict[str, int]
    spark_counts: dict[str, int]
    diffs: list[TableDiff]
    zoneinfo_problems: list[str]
    spot_checks: dict[str, pd.DataFrame]
    use_delta: bool

    @property
    def ok(self) -> bool:
        return all(d.ok for d in self.diffs) and not self.zoneinfo_problems


def validate(
    bronze_dir: Path,
    out: Path,
    *,
    windows=None,
    max_records: int = 200_000,
    timezone_name: str = DEFAULT_PARAMS["TIMEZONE"],
    use_delta: bool | None = None,
    spark=None,
    log=print,
) -> Report:
    """Sample -> DuckDB + Spark -> compare. Pass `spark` to reuse (and keep) a session."""
    import shutil

    import duckdb

    params = {"TIMEZONE": timezone_name}
    if out.exists():
        shutil.rmtree(out)
    sample_dir = out / "bronze"
    log(f"== sample -> {sample_dir}")
    windows = windows or default_windows(bronze_dir, timezone_name)
    sample_counts = make_sample(bronze_dir, sample_dir, windows, max_records, tz=timezone_name)
    switch_days = ", ".join(f"DATE'{date.fromisoformat(a) + timedelta(days=5)}'" for a, _ in windows[:-1]) or "NULL"
    log("   " + ", ".join(f"{k}={v:,}" for k, v in sample_counts.items()))

    log("== duckdb (sqlglot transpile)")
    db_path = out / "duckdb" / "sample.duckdb"
    duck_counts = build_layers(db_path, sample_dir, ("silver", "gold"), params)

    own_session = spark is None
    if own_session:
        spark, use_delta = spark_session(out / "spark-warehouse", use_delta)
    else:
        use_delta = bool(use_delta)
    log(
        f"== spark {spark.version} (ansi={spark.conf.get('spark.sql.ansi.enabled')}, "
        f"tz={spark.conf.get('spark.sql.session.timeZone')}, {'delta tables' if use_delta else 'cached temp views'})"
    )
    try:
        spark_counts = build_spark(spark, sample_dir, "table" if use_delta else "view", params)

        log("== compare")
        con = duckdb.connect(str(db_path), read_only=True)
        diffs = []
        for layer in ("silver", "gold"):
            for table in LAYERS[layer]:
                d = compare_table(table, con.execute(f"SELECT * FROM {table}").df(), spark.table(table).toPandas())
                diffs.append(d)
                log(f"   {'OK  ' if d.ok else 'DIFF'} {table:<28} duckdb={d.duck_rows:>9,} spark={d.spark_rows:>9,}")
                for p in d.problems:
                    log(f"        - {p}")
        zone_problems = check_against_zoneinfo(spark, timezone_name)
        log(f"   {'OK  ' if not zone_problems else 'DIFF'} local time vs Python zoneinfo (2,000 records)")
        for p in zone_problems[:5]:
            log(f"        - {p}")

        spot_sql = {
            "daily steps": "SELECT local_date, round(steps) AS steps, round(active_kcal) AS active_kcal FROM gold_daily_activity ORDER BY local_date",
            "resting HR": "SELECT local_date, round(resting_hr, 1) AS resting_hr, round(hrv_sdnn_ms, 1) AS hrv FROM gold_daily_vitals WHERE resting_hr IS NOT NULL ORDER BY local_date LIMIT 8",
            "DST hours": "SELECT local_date, local_hour, round(value) AS steps, chosen_source FROM gold_hourly_activity WHERE metric = 'StepCount' AND local_date IN ("
            + switch_days
            + ") AND local_hour BETWEEN 1 AND 4 ORDER BY 1, 2",
            "sleep nights": "SELECT night_date, round(asleep_min) AS asleep_min, round(in_bed_min) AS in_bed_min, round(bedtime_hours_after_18, 2) AS bed_h18 FROM gold_daily_sleep ORDER BY night_date LIMIT 8",
        }
        spot = {}
        for name, q in spot_sql.items():
            d, s = con.execute(q).df(), spark.sql(q).toPandas()
            keys = list(d.columns[: 2 if "hours" in name else 1])
            d[keys[0]], s[keys[0]] = d[keys[0]].astype(str), s[keys[0]].astype(str)
            spot[name] = d.merge(s, on=keys, suffixes=("_duckdb", "_spark"))
        con.close()
    finally:
        if own_session:
            spark.stop()
    return Report(sample_counts, duck_counts, spark_counts, diffs, zone_problems, spot, use_delta)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bronze-dir", type=Path, default=REPO / "data" / "bronze")
    p.add_argument("--out", type=Path, default=REPO / "data" / "_spark_test")
    p.add_argument("--max-records", type=int, default=200_000)
    p.add_argument("--timezone", default="auto", help="IANA zone or auto (most common HKTimeZone in bronze)")
    p.add_argument("--no-delta", action="store_true", help="use cached temp views instead of Delta tables")
    args = p.parse_args(argv)
    from health_lakehouse.timezone import resolve_timezone

    tz = resolve_timezone(args.timezone, args.bronze_dir)
    report = validate(
        args.bronze_dir,
        args.out,
        max_records=args.max_records,
        timezone_name=tz,
        use_delta=False if args.no_delta else None,
    )
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        for name, df in report.spot_checks.items():
            print(f"\n-- {name}\n{df.head(10).to_string(index=False)}")
    n_ok = sum(d.ok for d in report.diffs)
    print(
        f"\n{n_ok}/{len(report.diffs)} tables identical; zoneinfo check {'ok' if not report.zoneinfo_problems else 'FAILED'}"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
