"""Data quality expectations for silver/gold, written in Databricks SQL.

Each check is one query returning a single row with `value` (what was
measured) and `ok` (boolean). `error` checks fail the Databricks job,
`warn` checks are only reported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .pipeline import LAYERS

# Must never be empty for a real export; the remaining tables may legitimately be
# empty (no ECG, no GPS routes, no clinical records).
REQUIRED_TABLES = [
    "silver_records",
    "silver_activity_summaries",
    "gold_hourly_activity",
    "gold_daily_activity",
    "gold_daily_vitals",
    "gold_daily_summary",
    "gold_weekly_summary",
]


@dataclass(frozen=True)
class Check:
    name: str
    sql: str
    severity: str = "error"  # "error" | "warn"


@dataclass
class CheckResult:
    name: str
    severity: str
    ok: bool
    value: str


def _count_check(table: str) -> Check:
    severity = "error" if table in REQUIRED_TABLES else "warn"
    return Check(f"{table} has rows", f"SELECT COUNT(*) AS value, COUNT(*) > 0 AS ok FROM {table}", severity)


def _unique_check(table: str, keys: str) -> Check:
    return Check(
        f"{table} unique on ({keys})",
        f"SELECT COUNT(*) AS value, COUNT(*) = 0 AS ok FROM "
        f"(SELECT {keys} FROM {table} GROUP BY {keys} HAVING COUNT(*) > 1) d",
    )


CHECKS: list[Check] = [
    *[_count_check(t) for layer in ("silver", "gold") for t in LAYERS[layer]],
    _unique_check("silver_records", "record_id"),
    _unique_check("gold_hourly_activity", "local_date, local_hour, metric"),
    _unique_check("gold_daily_summary", "local_date"),
    _unique_check("gold_daily_sleep", "night_date"),
    _unique_check("gold_workouts", "workout_id"),
    Check(
        "silver_records local_date/local_hour populated",
        "SELECT SUM(CASE WHEN local_date IS NULL OR local_hour IS NULL THEN 1 ELSE 0 END) AS value, "
        "SUM(CASE WHEN local_date IS NULL OR local_hour IS NULL THEN 1 ELSE 0 END) <= 0.001 * COUNT(*) AS ok "
        "FROM silver_records",
    ),
    Check(
        "silver_records local_hour in 0..23",
        "SELECT COUNT(*) AS value, COUNT(*) = 0 AS ok FROM silver_records WHERE local_hour NOT BETWEEN 0 AND 23",
    ),
    Check(
        "silver_records dates between 2000-01-01 and tomorrow",
        "SELECT concat(CAST(MIN(local_date) AS STRING), ' .. ', CAST(MAX(local_date) AS STRING)) AS value, "
        "MIN(local_date) >= DATE'2000-01-01' AND MAX(local_date) <= date_add(current_date(), 1) AS ok "
        "FROM silver_records",
    ),
    Check(
        "silver_records end >= start",
        "SELECT COUNT(*) AS value, COUNT(*) <= 0.001 * (SELECT COUNT(*) FROM silver_records) AS ok "
        "FROM silver_records WHERE end_utc < start_utc",
    ),
    Check(
        "gold_daily_summary covers every day of silver_records",
        "SELECT (SELECT COUNT(*) FROM gold_daily_summary) AS value, "
        "(SELECT COUNT(*) FROM gold_daily_summary) = (SELECT COUNT(DISTINCT local_date) FROM silver_records) AS ok",
    ),
    Check(
        "share of days with steps >= 50%",
        "SELECT round(100 * AVG(CASE WHEN steps > 0 THEN 1 ELSE 0 END), 1) AS value, "
        "AVG(CASE WHEN steps > 0 THEN 1 ELSE 0 END) >= 0.5 AS ok FROM gold_daily_summary",
        "warn",
    ),
    Check(
        "daily steps <= 100k",
        "SELECT MAX(steps) AS value, coalesce(MAX(steps), 0) <= 100000 AS ok FROM gold_daily_activity",
    ),
    # Real exports contain the odd implausible sample (e.g. one day with resting HR 158),
    # so plausibility checks allow a tiny share of outliers instead of requiring zero.
    Check(
        "resting HR within 25..150 bpm on >= 99% of days",
        "SELECT SUM(CASE WHEN resting_hr NOT BETWEEN 25 AND 150 THEN 1 ELSE 0 END) AS value, "
        "coalesce(AVG(CASE WHEN resting_hr NOT BETWEEN 25 AND 150 THEN 1 ELSE 0 END), 0) <= 0.01 AS ok "
        "FROM gold_daily_vitals WHERE resting_hr IS NOT NULL",
    ),
    Check(
        "sleep per night <= 24 h",
        "SELECT MAX(asleep_min) AS value, coalesce(MAX(asleep_min), 0) <= 24 * 60 AS ok FROM gold_daily_sleep",
    ),
    Check(
        "hourly steps not double counted (> 30k steps in <= 0.1% of hours)",
        "SELECT SUM(CASE WHEN value > 30000 THEN 1 ELSE 0 END) AS value, "
        "coalesce(AVG(CASE WHEN value > 30000 THEN 1 ELSE 0 END), 0) <= 0.001 AS ok "
        "FROM gold_hourly_activity WHERE metric = 'StepCount'",
    ),
]


def run_checks(query: Callable[[str], tuple], checks: list[Check] = CHECKS) -> list[CheckResult]:
    """`query(sql)` must return the first row as a `(value, ok)` tuple."""
    results = []
    for check in checks:
        value, ok = query(check.sql)
        results.append(CheckResult(check.name, check.severity, bool(ok), str(value)))
    return results
