"""Run the Databricks (Spark SQL) transformation files locally on DuckDB.

The SQL files under `sql/` are the single source of truth and are written in
Databricks SQL. Locally they are transpiled with sqlglot. sqlglot handles
almost everything; the few Spark functions whose semantics it cannot map
are renamed to `spark_*` before parsing and implemented as DuckDB macros.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

# Spark function -> DuckDB macro with identical semantics.
_RENAMES = {
    # sqlglot maps this to `AT TIME ZONE`, which on a naive DuckDB TIMESTAMP
    # means "interpret as local", the opposite of Spark's "interpret as UTC".
    "from_utc_timestamp": "spark_from_utc_timestamp",
    # Spark: Sunday=1..Saturday=7; DuckDB: Sunday=0..Saturday=6.
    "dayofweek": "spark_dayofweek",
    # Not transpiled by sqlglot. (`try_cast(x AS TIMESTAMP)` would work in both
    # engines but becomes TIMESTAMPTZ in DuckDB.)
    "try_to_timestamp": "spark_try_to_timestamp",
}
_RENAME_RE = re.compile(r"\b(" + "|".join(_RENAMES) + r")\s*\(", re.IGNORECASE)

DUCKDB_MACROS = [
    "CREATE OR REPLACE MACRO spark_from_utc_timestamp(ts, tz) AS timezone(tz, timezone('UTC', CAST(ts AS TIMESTAMP)))",
    "CREATE OR REPLACE MACRO spark_dayofweek(d) AS dayofweek(CAST(d AS DATE)) + 1",
    "CREATE OR REPLACE MACRO spark_try_to_timestamp(s) AS TRY_CAST(s AS TIMESTAMP)",
]


def render(sql: str, params: dict[str, str]) -> str:
    """Substitute `{{NAME}}` placeholders (shared by local + Databricks runners)."""
    for key, value in params.items():
        sql = sql.replace("{{" + key + "}}", value)
    return sql


def to_duckdb(spark_sql: str) -> str:
    import sqlglot

    renamed = _RENAME_RE.sub(lambda m: _RENAMES[m.group(1).lower()] + "(", spark_sql)
    statements = sqlglot.transpile(renamed, read="databricks", write="duckdb")
    if len(statements) != 1:
        raise ValueError(f"expected exactly one statement, got {len(statements)}")
    return statements[0]


def prepare_connection(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET TimeZone = 'UTC'")
    for macro in DUCKDB_MACROS:
        con.execute(macro)
