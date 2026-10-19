"""Unit tests for the DuckDB compatibility layer and the SQL manifest."""

from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import duckdb
import pytest
import sqlglot
from sqlglot import exp

from health_lakehouse.pipeline import DEFAULT_PARAMS, LAYERS, load_sql
from health_lakehouse.spark_runner import table_comment
from health_lakehouse.sql_compat import prepare_connection, render, to_duckdb

REPO = Path(__file__).resolve().parents[1]
MANIFEST = [(layer, table) for layer, tables in LAYERS.items() for table in tables]


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    prepare_connection(c)
    yield c
    c.close()


def run(con, spark_sql: str):
    return con.execute(to_duckdb(spark_sql)).fetchone()


@pytest.mark.parametrize(
    "utc, expected",
    [
        ("2024-01-15 07:00:00", datetime(2024, 1, 15, 8, 0)),  # CET, UTC+1
        ("2024-07-15 07:00:00", datetime(2024, 7, 15, 9, 0)),  # CEST, UTC+2
        ("2024-03-31 00:59:59", datetime(2024, 3, 31, 1, 59, 59)),  # just before spring forward
        ("2024-03-31 01:00:00", datetime(2024, 3, 31, 3, 0)),  # 02:00 local is skipped
        ("2024-10-27 00:30:00", datetime(2024, 10, 27, 2, 30)),  # first 02:30 (CEST)
        ("2024-10-27 01:30:00", datetime(2024, 10, 27, 2, 30)),  # second 02:30 (CET)
    ],
)
def test_from_utc_timestamp_follows_dst(con, utc, expected):
    assert run(con, f"SELECT from_utc_timestamp(TIMESTAMP'{utc}', 'Europe/Berlin')")[0] == expected


def test_from_utc_timestamp_independent_of_session_timezone():
    c = duckdb.connect()
    prepare_connection(c)
    c.execute("SET TimeZone = 'America/New_York'")
    sql = "SELECT from_utc_timestamp(TIMESTAMP'2024-01-15 07:00:00', 'Europe/Berlin')"
    assert c.execute(to_duckdb(sql)).fetchone()[0] == datetime(2024, 1, 15, 8, 0)


@pytest.mark.parametrize(
    "day, expected",
    [("2024-01-14", 1), ("2024-01-15", 2), ("2024-01-19", 6), ("2024-01-20", 7)],  # Sun=1 .. Sat=7
)
def test_dayofweek_is_spark_numbering(con, day, expected):
    assert run(con, f"SELECT dayofweek(DATE'{day}')")[0] == expected


def test_week_starts_monday(con):
    assert run(con, "SELECT to_date(date_trunc('WEEK', DATE'2024-01-14'))")[0] == date(2024, 1, 8)


def test_json_path_with_spaces(con):
    doc = """'{"Recorded Date": "2024-01-15 08:00:00 +0100"}'"""
    assert run(con, f"SELECT get_json_object({doc}, '$.Recorded Date')")[0] == "2024-01-15 08:00:00 +0100"
    assert run(con, f"""SELECT get_json_object({doc}, "$['Recorded Date']")""")[0] == "2024-01-15 08:00:00 +0100"


def test_division_and_try_cast(con):
    assert run(con, "SELECT 90 / 60")[0] == 1.5  # Spark `/` is never integer division
    assert run(con, "SELECT try_cast('abc' AS INT)")[0] is None
    assert run(con, "SELECT try_cast('2024-01-15 08:00:00' AS TIMESTAMP_NTZ)")[0] == datetime(2024, 1, 15, 8)
    assert run(con, "SELECT 5 / nullif(0, 0)")[0] is None


def test_try_to_timestamp_is_naive_and_tolerant(con):
    assert run(con, "SELECT try_to_timestamp('2024-01-15 08:00:00')")[0] == datetime(2024, 1, 15, 8)
    assert run(con, "SELECT try_to_timestamp('2024-13-45 99:00:00')")[0] is None


def test_timestampadd_and_diff(con):
    wall = "try_to_timestamp('2024-01-15 01:00:00')"
    assert run(con, f"SELECT timestampadd(MINUTE, -120, {wall})")[0] == datetime(2024, 1, 14, 23)
    assert (
        run(con, "SELECT timestampdiff(SECOND, TIMESTAMP'2024-01-15 00:00:00', TIMESTAMP'2024-01-15 01:00:30')")[0]
        == 3630
    )


def test_render_replaces_placeholders():
    assert render("SELECT '{{TIMEZONE}}', '{{TIMEZONE}}'", {"TIMEZONE": "UTC"}) == "SELECT 'UTC', 'UTC'"


@pytest.mark.parametrize("layer, table", MANIFEST, ids=[t for _, t in MANIFEST])
def test_manifest_sql_parses_as_databricks(layer, table):
    sql = render(load_sql(layer, table), DEFAULT_PARAMS)
    assert "{{" not in sql
    statements = sqlglot.parse(sql, read="databricks")
    assert len(statements) == 1
    assert isinstance(statements[0], (exp.Select, exp.Union)), f"{table} must be a single SELECT"


@pytest.mark.parametrize("layer, table", MANIFEST, ids=[t for _, t in MANIFEST])
def test_manifest_sql_transpiles_to_duckdb(con, layer, table):
    duck_sql = to_duckdb(render(load_sql(layer, table), DEFAULT_PARAMS))
    assert isinstance(sqlglot.parse_one(duck_sql, read="duckdb"), (exp.Select, exp.Union))
    for fn in ("from_utc_timestamp", "dayofweek", "try_to_timestamp"):
        assert f" {fn}(" not in duck_sql.lower().replace(f"spark_{fn}(", ""), fn


@pytest.mark.parametrize("layer, table", MANIFEST, ids=[t for _, t in MANIFEST])
def test_manifest_sql_has_table_comment(layer, table):
    assert len(table_comment(load_sql(layer, table))) > 10


def test_manifest_files_exist_and_are_complete():
    sql_dir = REPO / "src" / "health_lakehouse" / "sql"
    on_disk = {(p.parent.name, p.stem) for p in sql_dir.glob("*/*.sql")}
    assert on_disk == set(MANIFEST)


def test_manifest_order_respects_dependencies():
    """A table may only reference bronze tables or tables earlier in the manifest."""
    seen: set[str] = set()
    for layer, table in MANIFEST:
        tree = sqlglot.parse_one(render(load_sql(layer, table), DEFAULT_PARAMS), read="databricks")
        ctes = {c.alias for c in tree.find_all(exp.CTE)}
        for ref in {t.name for t in tree.find_all(exp.Table)} - ctes:
            assert ref.startswith("bronze_") or ref in seen, f"{table} reads {ref} before it is built"
        seen.add(table)


def test_pipeline_imports_without_duckdb_or_sqlglot():
    """Databricks puts only `src/` on sys.path; the manifest must not need DuckDB/sqlglot."""
    code = (
        "import builtins, sys\n"
        "real = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] in ('duckdb', 'sqlglot'):\n"
        "        raise ImportError('blocked ' + name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
        "from health_lakehouse.pipeline import LAYERS, load_sql, render, DEFAULT_PARAMS\n"
        "from health_lakehouse.spark_runner import build_layer_spark, table_comment\n"
        "print(len(render(load_sql('gold', LAYERS['gold'][-1]), DEFAULT_PARAMS)))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd="/")
    assert out.returncode == 0, out.stderr
    assert int(out.stdout) > 0


def test_data_quality_checks_transpile():
    from health_lakehouse.quality import CHECKS

    for check in CHECKS:
        assert isinstance(sqlglot.parse_one(to_duckdb(check.sql), read="duckdb"), exp.Select), check.name
