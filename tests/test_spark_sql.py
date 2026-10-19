"""Run the Databricks SQL on a local Spark and compare it with the DuckDB path.

Runs on the synthetic bronze data from `conftest.py`. Skipped automatically when
PySpark or a Java 17/21 runtime is missing. See scripts/validate_spark_sql.py for
the full report on your own export.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytest.importorskip("pyspark")

_spec = importlib.util.spec_from_file_location("validate_spark_sql", REPO / "scripts" / "validate_spark_sql.py")
validate_spark_sql = importlib.util.module_from_spec(_spec)
sys.modules["validate_spark_sql"] = validate_spark_sql
_spec.loader.exec_module(validate_spark_sql)

if validate_spark_sql.find_java_home() is None:
    pytest.skip("no Java 17/21 (set JAVA_HOME or unpack a JDK into .cache/jdk/)", allow_module_level=True)

from health_lakehouse.pipeline import LAYERS

TABLES = [t for layer in ("silver", "gold") for t in LAYERS[layer]]


@pytest.fixture(scope="module")
def spark_and_delta(tmp_path_factory):
    spark, use_delta = validate_spark_sql.spark_session(tmp_path_factory.mktemp("warehouse"))
    yield spark, use_delta
    spark.stop()


@pytest.fixture(scope="module")
def spark(spark_and_delta):
    return spark_and_delta[0]


@pytest.fixture(scope="module")
def report(spark_and_delta, synthetic_bronze, tmp_path_factory):
    spark, use_delta = spark_and_delta
    return validate_spark_sql.validate(
        synthetic_bronze,
        tmp_path_factory.mktemp("spark_test"),
        max_records=60_000,
        spark=spark,
        use_delta=use_delta,
        log=lambda *_: None,
    )


def one(spark, sql):
    return spark.sql(sql).collect()[0][0]


def test_session_is_configured_like_databricks(spark):
    assert spark.conf.get("spark.sql.session.timeZone") == "UTC"
    assert spark.conf.get("spark.sql.ansi.enabled") == "true"


@pytest.mark.parametrize(
    "utc, expected",
    [("2024-01-15 07:00:00", datetime(2024, 1, 15, 8)), ("2024-07-15 07:00:00", datetime(2024, 7, 15, 9))],
)
def test_spark_from_utc_timestamp(spark, utc, expected):
    sql = f"SELECT CAST(from_utc_timestamp(TIMESTAMP'{utc}', 'Europe/Berlin') AS TIMESTAMP_NTZ)"
    assert one(spark, sql) == expected


def test_spark_function_semantics(spark):
    assert one(spark, "SELECT dayofweek(DATE'2024-01-14')") == 1
    assert one(spark, "SELECT to_date(date_trunc('WEEK', DATE'2024-01-14'))") == date(2024, 1, 8)
    doc = """'{"Recorded Date": "x"}'"""
    assert one(spark, f"SELECT get_json_object({doc}, '$.Recorded Date')") == "x"
    assert one(spark, f"""SELECT get_json_object({doc}, "$['Recorded Date']")""") == "x"
    assert one(spark, "SELECT try_to_timestamp('2024-13-45 99:00:00')") is None
    assert one(spark, "SELECT 90 / 60") == 1.5
    assert one(spark, "SELECT try_cast('abc' AS INT)") is None


@pytest.mark.parametrize("table", TABLES)
def test_table_matches_duckdb(report, table):
    diff = next(d for d in report.diffs if d.table == table)
    assert diff.ok, "\n".join(diff.problems)


def test_core_tables_not_empty(report):
    for table in ("silver_records", "gold_hourly_activity", "gold_daily_activity", "gold_daily_summary"):
        assert report.spark_counts[table] > 0, table


def test_local_time_matches_zoneinfo(report):
    assert not report.zoneinfo_problems, report.zoneinfo_problems[:5]


def test_sample_windows_cover_dst_switches():
    from datetime import date

    windows = validate_spark_sql.dst_windows(date(2025, 8, 1), date(2026, 9, 30), "Europe/Berlin")
    assert windows[:2] == [("2025-10-21", "2025-10-31"), ("2026-03-24", "2026-04-03")]
    assert windows[-1] == ("2026-09-17", "2026-09-30")
    assert validate_spark_sql.dst_windows(date(2025, 1, 1), date(2025, 12, 31), "UTC") == [("2025-12-18", "2025-12-31")]


def test_comparison_detects_differences():
    import pandas as pd

    a = pd.DataFrame({"local_date": ["2024-01-01", "2024-01-02"], "steps": [100.0, 200.0]})
    b = a.assign(steps=[100.0, 200.5])
    assert validate_spark_sql.compare_table("gold_daily_activity", a, a.copy()).ok
    assert not validate_spark_sql.compare_table("gold_daily_activity", a, b).ok


def test_data_quality_checks_run_on_spark(report, spark):
    from health_lakehouse.quality import run_checks

    results = run_checks(lambda sql: tuple(spark.sql(sql).collect()[0]))
    failed = [r for r in results if not r.ok and r.severity == "error"]
    assert not failed, failed


# Last: replaces the bronze_* views the report tables were built from.
def test_timezone_auto_detection_on_spark(spark, synthetic_bronze):
    from health_lakehouse.timezone import resolve_timezone_spark

    for t in ("workouts", "records"):
        spark.read.parquet(str(synthetic_bronze / f"{t}.parquet")).createOrReplaceTempView(f"bronze_{t}")
    assert resolve_timezone_spark(spark, "auto", log=lambda *_: None) == "Europe/Berlin"
    assert resolve_timezone_spark(spark, "Asia/Tokyo") == "Asia/Tokyo"
