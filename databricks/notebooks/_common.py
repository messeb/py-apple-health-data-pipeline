# Databricks notebook source
# MAGIC %md
# MAGIC # Shared setup (included with `%run ./_common`)
# MAGIC
# MAGIC * Widgets: `catalog`, `schema`, `volume`, `timezone` (the job passes them as parameters).
# MAGIC * Puts the repo's `src/` on `sys.path`, so `health_lakehouse` can be imported without
# MAGIC   pip-installing it. Works for bundle deployments (`.bundle/<name>/<target>/files/...`)
# MAGIC   and for Git folders.

# COMMAND ----------

import os
import sys

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "health", "Schema")
dbutils.widgets.text("volume", "raw", "Volume")
dbutils.widgets.text("timezone", "auto", "Timezone (IANA or auto)")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
VOLUME = dbutils.widgets.get("volume").strip()
TIMEZONE = dbutils.widgets.get("timezone").strip()

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
EXPORT_DIR = f"{VOLUME_ROOT}/apple_health_export"
FQ = f"`{CATALOG}`.`{SCHEMA}`"

# COMMAND ----------


def _repo_root() -> str:
    """Folder that contains `src/` and `databricks/`, found from this notebook's location."""
    candidates = [os.getcwd()]  # the notebook's folder on DBR 14+ / serverless
    try:
        nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        candidates.append(nb_path if nb_path.startswith("/Workspace") else "/Workspace" + nb_path)
    except Exception:
        pass
    for path in candidates:
        if "/databricks/notebooks" in path:
            root = path.split("/databricks/notebooks")[0]
            if os.path.isdir(os.path.join(root, "src", "health_lakehouse")):
                return root
    raise RuntimeError(f"could not locate the repo root from {candidates}")


REPO_ROOT = _repo_root()
if f"{REPO_ROOT}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_ROOT}/src")

# Wall-clock strings are parsed as UTC; all local times come from from_utc_timestamp(..., TIMEZONE).
spark.conf.set("spark.sql.session.timeZone", "UTC")

# `auto`: most common HKTimeZone in bronze_workouts/bronze_records. Resolved here once, so silver
# and gold always agree. Before bronze exists (01_bronze, which needs no zone) it stays `auto`.
if TIMEZONE.lower() == "auto" and spark.catalog.tableExists(f"{CATALOG}.{SCHEMA}.bronze_records"):
    from health_lakehouse.timezone import resolve_timezone_spark

    TIMEZONE = resolve_timezone_spark(spark, TIMEZONE, schema=FQ)

print(f"catalog={CATALOG} schema={SCHEMA} volume={VOLUME} timezone={TIMEZONE}")
print(f"repo={REPO_ROOT}")
