"""Pick the IANA timezone used for local dates in silver (`--timezone auto`).

Apple stamps every timestamp with the UTC offset active at export time, so the
offset alone cannot tell CET from CEST. Workouts (and many third-party records)
carry an `HKTimeZone` metadata entry with the real IANA name; the most common
one is the person's home zone. Without any, fall back to the machine's zone,
then UTC.

`TIMEZONE_COUNTS_SQL` is Databricks SQL like the layer files, so the local
DuckDB runner and the Databricks silver notebook detect the zone the same way.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = [
    "TIMEZONE_COUNTS_SQL",
    "detect_from_bronze",
    "is_iana",
    "pick_timezone",
    "resolve_timezone",
    "resolve_timezone_spark",
    "system_timezone",
]

AUTO = "auto"

TIMEZONE_COUNTS_SQL = """
SELECT tz, COUNT(*) AS n
FROM (
  SELECT get_json_object(metadata_json, '$.HKTimeZone') AS tz FROM bronze_workouts
  WHERE metadata_json LIKE '%HKTimeZone%'
  UNION ALL
  SELECT get_json_object(metadata_json, '$.HKTimeZone') AS tz FROM bronze_records
  WHERE metadata_json LIKE '%HKTimeZone%'
) t
WHERE tz IS NOT NULL
GROUP BY tz
ORDER BY n DESC, tz
""".strip()


def is_iana(name: str | None) -> bool:
    """True for a region name like 'Europe/Berlin' (or 'UTC'); False for abbreviations such as 'CET'."""
    if not name or name != name.strip():
        return False
    if "/" not in name and name not in ("UTC", "Etc/UTC"):
        return False
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


def pick_timezone(counts: Iterable[tuple[str | None, int]]) -> tuple[str, int, int] | None:
    """Most common valid zone from (name, count) pairs: (zone, its count, total valid count)."""
    totals: dict[str, int] = {}
    for name, n in counts:
        if is_iana(name):
            totals[name] = totals.get(name, 0) + int(n)
    if not totals:
        return None
    best = min(totals, key=lambda z: (-totals[z], z))
    return best, totals[best], sum(totals.values())


def system_timezone() -> str | None:
    """IANA name of this machine's zone, if it can be determined (TZ, /etc/localtime, /etc/timezone)."""
    tz = os.environ.get("TZ", "").lstrip(":")
    if is_iana(tz):
        return tz
    try:
        target = os.path.realpath("/etc/localtime")
        if "zoneinfo/" in target:
            name = target.split("zoneinfo/", 1)[1]
            if is_iana(name):
                return name
    except OSError:
        pass
    try:
        name = Path("/etc/timezone").read_text().strip()
        if is_iana(name):
            return name
    except OSError:
        pass
    return None


def detect_from_bronze(bronze_dir: str | Path) -> tuple[str, int, int] | None:
    """Count HKTimeZone values in the local bronze Parquet files with DuckDB."""
    import duckdb

    from .sql_compat import prepare_connection, to_duckdb

    bronze_dir = Path(bronze_dir)
    con = duckdb.connect()
    try:
        prepare_connection(con)
        for t in ("workouts", "records"):
            path = bronze_dir / f"{t}.parquet"
            src = (
                f"read_parquet('{path.as_posix()}')"
                if path.exists()
                else "(SELECT CAST(NULL AS VARCHAR) AS metadata_json WHERE false)"
            )
            con.execute(f"CREATE OR REPLACE VIEW bronze_{t} AS SELECT metadata_json FROM {src}")
        rows = con.execute(to_duckdb(TIMEZONE_COUNTS_SQL)).fetchall()
    finally:
        con.close()
    return pick_timezone(rows)


def resolve_timezone(requested: str, bronze_dir: str | Path | None = None, log=print) -> str:
    """Return an IANA zone: `requested` itself, or for 'auto' the detected/system/UTC zone."""
    return _resolve(requested, lambda: detect_from_bronze(bronze_dir) if bronze_dir else None, system_timezone, log)


def resolve_timezone_spark(spark, requested: str, schema: str | None = None, log=print) -> str:
    """Same as `resolve_timezone`, counting in the bronze tables of `schema` (default: current) with Spark SQL.

    Without bronze tables yet (the bronze task itself) 'auto' stays undecided and UTC is returned;
    on Databricks the machine zone says nothing about the person, so UTC is also the fallback.
    """
    prefix = f"{schema}." if schema else ""

    def detect():
        if not all(spark.catalog.tableExists(f"{prefix}bronze_{t}") for t in ("workouts", "records")):
            return None
        sql = TIMEZONE_COUNTS_SQL.replace("FROM bronze_", f"FROM {prefix}bronze_")
        return pick_timezone((r[0], r[1]) for r in spark.sql(sql).collect())

    return _resolve(requested, detect, lambda: None, log)


def _resolve(requested: str, detect, system, log) -> str:
    requested = (requested or AUTO).strip()
    if requested.lower() != AUTO:
        if not is_iana(requested):
            raise ValueError(f"unknown IANA timezone {requested!r} (use e.g. 'Europe/Berlin' or 'auto')")
        return requested
    found = detect()
    if found:
        tz, n, total = found
        log(f"timezone: {tz} (auto: most common HKTimeZone, {n:,} of {total:,} tagged samples)")
        return tz
    tz = system()
    if tz:
        log(f"timezone: {tz} (auto: no HKTimeZone metadata in bronze, using this machine's zone)")
        return tz
    log("timezone: UTC (auto: no HKTimeZone metadata and no resolvable system zone)")
    return "UTC"
