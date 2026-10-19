"""The Lakeview dashboard (databricks/dashboards/health.lvdash.json) only queries real gold columns.

Every dataset query, and every dimension/measure expression on top of it, runs on DuckDB against
the synthetic gold tables, so a renamed column breaks this test instead of the deployed dashboard.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import pytest

from health_lakehouse.sql_compat import prepare_connection, to_duckdb

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "databricks" / "dashboards" / "health.lvdash.json"


def _datasets() -> list[dict]:
    return json.loads(DASHBOARD.read_text())["datasets"]


def _source_sql(ds: dict) -> str:
    if "queryLines" in ds:
        return "".join(ds["queryLines"])
    source = ds["config"]["source"].strip()
    return source if re.match(r"(?is)^\s*select\b", source) else f"SELECT * FROM {source}"


def test_dashboard_is_portable():
    text = DASHBOARD.read_text()
    # Catalog and schema come from the bundle (dataset_catalog / dataset_schema), never the file.
    assert "workspace." not in text and "apple_health" not in text
    for ds in _datasets():
        assert "catalog" not in ds and "schema" not in ds, ds["name"]


@pytest.fixture(scope="module")
def con(synthetic_root):
    con = duckdb.connect()
    prepare_connection(con)
    for parquet in sorted((synthetic_root / "data" / "gold").glob("*.parquet")):
        con.execute(f"CREATE VIEW {parquet.stem} AS SELECT * FROM read_parquet('{parquet}')")
    return con


@pytest.mark.parametrize("ds", _datasets(), ids=lambda ds: ds["name"])
def test_dataset_queries_run_on_gold(con, ds):
    source = _source_sql(ds)
    con.execute(to_duckdb(f"SELECT * FROM ({source}) LIMIT 5")).fetchall()
    config = ds.get("config", {})
    for dim in config.get("dimensions", []):
        con.execute(to_duckdb(f"SELECT {dim['expr']} FROM ({source}) t LIMIT 5")).fetchall()
    for measure in config.get("measures", []):
        con.execute(to_duckdb(f"SELECT {measure['expr']} FROM ({source}) t")).fetchall()
