# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze: parse the raw export into Delta tables
# MAGIC
# MAGIC `export.xml` is a single multi-GB XML document, so Spark cannot split it. It is parsed on the
# MAGIC driver with `lxml.iterparse` (the same `build_bronze()` as locally) into Parquet on local disk,
# MAGIC the Parquet files are staged in the volume (executors can't see driver-local disk), and every
# MAGIC file becomes a Delta table `bronze_<name>`. All columns stay strings; typing happens in silver.
# MAGIC Re-running overwrites the tables, so the notebook is idempotent.

# COMMAND ----------

# MAGIC %pip install -q "lxml>=5.2"

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import os
import shutil
import tempfile
import time

from health_lakehouse.bronze import build_bronze
from health_lakehouse.pipeline import BRONZE_TABLES

BRONZE_COMMENTS = {
    "records": "Raw HealthKit <Record> samples from export.xml (quantity + category)",
    "hrv_instantaneous_bpm": "Beat-to-beat BPM lists attached to HRV records",
    "correlations": "Raw <Correlation> elements (e.g. blood pressure pairs)",
    "workouts": "Raw <Workout> elements",
    "workout_events": "Raw <WorkoutEvent> markers (pause, lap, segment)",
    "workout_statistics": "Raw <WorkoutStatistics> per workout",
    "activity_summaries": "Raw <ActivitySummary> (daily rings)",
    "me": "Raw <Me> characteristics (date of birth, sex, blood type)",
    "ecg_readings": "ECG CSV headers (name and birth date dropped)",
    "ecg_samples": "ECG CSV voltage samples",
    "route_points": "GPX workout route track points",
    "clinical_records": "FHIR clinical records (raw JSON)",
}


def _writable_dir(candidates):
    # Classic clusters have a fast /local_disk0; on serverless it exists but is read-only.
    for root in candidates:
        try:
            probe = os.path.join(root, "health_bronze")
            os.makedirs(probe, exist_ok=True)
            return root
        except OSError:
            continue
    raise RuntimeError(f"no writable temp directory among {candidates}")


local_root = _writable_dir(["/local_disk0/tmp", tempfile.gettempdir(), os.path.expanduser("~")])
print("local scratch dir:", local_root)
local_bronze = os.path.join(local_root, "health_bronze")
staging = f"{VOLUME_ROOT}/_bronze_staging"
for path in (local_bronze, staging):
    shutil.rmtree(path, ignore_errors=True)
os.makedirs(local_bronze)

# COMMAND ----------

t0 = time.time()
counts = build_bronze(EXPORT_DIR, local_bronze)
print(f"parsed in {time.time() - t0:.0f}s:", counts)

# COMMAND ----------

os.makedirs(staging, exist_ok=True)
for name in BRONZE_TABLES:
    src = os.path.join(local_bronze, f"{name}.parquet")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} was not produced; is {EXPORT_DIR} complete?")
    shutil.copyfile(src, f"{staging}/{name}.parquet")

for name in BRONZE_TABLES:
    table = f"{FQ}.bronze_{name}"
    (
        spark.read.parquet(f"{staging}/{name}.parquet")
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table)
    )
    spark.sql(f"COMMENT ON TABLE {table} IS '{BRONZE_COMMENTS[name]} (bronze, all columns string)'")
    print(f"  bronze_{name:<24} {spark.table(table).count():>12,} rows")

# COMMAND ----------

shutil.rmtree(staging, ignore_errors=True)
shutil.rmtree(local_bronze, ignore_errors=True)
