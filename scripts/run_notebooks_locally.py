"""Execute the Databricks notebooks 01-04 on a local Spark + Delta session.

A smoke test for the notebook code itself (widgets, sys.path discovery,
bronze staging, CREATE OR REPLACE TABLE, comments, OPTIMIZE, data quality),
with a stub `dbutils`. Unity Catalog is replaced by `spark_catalog`, and the
volume by a local folder; 00_setup (CREATE VOLUME, unzip) is not run.

    .venv/bin/python scripts/run_notebooks_locally.py [--export-dir apple_health_export] [--work data/_spark_test/notebooks]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "databricks" / "notebooks"


class _Widgets:
    def __init__(self, values: dict[str, str]):
        self.values = values

    def text(self, name, default, label=None):
        self.values.setdefault(name, default)

    def dropdown(self, name, default, choices, label=None):
        self.values.setdefault(name, default)

    def get(self, name):
        return self.values[name]


class _DBUtils:
    """Just enough of `dbutils` for the pipeline notebooks."""

    def __init__(self, widgets: dict[str, str]):
        self.widgets = _Widgets(widgets)

    class notebook:
        class entry_point:
            @staticmethod
            def getDbutils():
                raise RuntimeError("no notebook context outside Databricks")


def run_notebook(name: str, ns: dict, after_common=None) -> None:
    print(f"\n######## {name}", flush=True)
    t0 = time.time()
    for cell in (NOTEBOOKS / name).read_text().split("# COMMAND ----------"):
        body = cell.replace("# Databricks notebook source", "").strip()
        if not body:
            continue
        if all(line.startswith("# MAGIC") for line in body.splitlines()):  # %md, %pip, %run
            if "%run ./_common" in body:
                exec(compile((NOTEBOOKS / "_common.py").read_text(), "_common.py", "exec"), ns)
                if after_common:
                    after_common(ns)
            continue
        exec(compile(body, name, "exec"), ns)
    print(f"######## {name} done in {time.time() - t0:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export-dir", type=Path, default=REPO / "apple_health_export")
    p.add_argument("--work", type=Path, default=REPO / "data" / "_spark_test" / "notebooks")
    p.add_argument("--timezone", default="auto", help="IANA zone or auto (detect from HKTimeZone)")
    args = p.parse_args(argv)

    spec = importlib.util.spec_from_file_location("validate_spark_sql", REPO / "scripts" / "validate_spark_sql.py")
    validate = importlib.util.module_from_spec(spec)
    sys.modules["validate_spark_sql"] = validate
    spec.loader.exec_module(validate)

    shutil.rmtree(args.work, ignore_errors=True)
    spark, _ = validate.spark_session(args.work / "warehouse", use_delta=True)
    widgets = {"catalog": "spark_catalog", "schema": "health", "timezone": args.timezone}
    ns = {"spark": spark, "dbutils": _DBUtils(widgets), "display": lambda df: df.show(60, truncate=False)}
    spark.sql("CREATE SCHEMA IF NOT EXISTS spark_catalog.health")  # 00_setup's job on Databricks

    def local_volume(ns: dict) -> None:
        ns["VOLUME_ROOT"] = str(args.work / "volume")
        ns["EXPORT_DIR"] = str(args.export_dir.resolve())
        os.makedirs(ns["VOLUME_ROOT"], exist_ok=True)

    os.chdir(NOTEBOOKS)  # like DBR 14+: cwd = the notebook's folder
    try:
        run_notebook("01_bronze.py", ns, local_volume)
        for name in ("02_silver.py", "03_gold.py", "04_data_quality.py"):
            run_notebook(name, ns)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
