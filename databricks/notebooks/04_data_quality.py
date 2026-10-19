# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Data quality
# MAGIC
# MAGIC Runs the expectations from `health_lakehouse.quality` (row counts, unique keys, date ranges,
# MAGIC plausibility of steps / heart rate / sleep). Results are appended to `data_quality_results`.
# MAGIC Any failed `error` check fails the notebook and therefore the job; `warn` checks are only
# MAGIC reported.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

from datetime import datetime, UTC

from health_lakehouse.quality import run_checks

spark.sql(f"USE {FQ}")  # = USE CATALOG + USE SCHEMA

results = run_checks(lambda sql: tuple(spark.sql(sql).collect()[0]))
run_at = datetime.now(UTC)
df = spark.createDataFrame(
    [(run_at, r.name, r.severity, r.ok, r.value) for r in results],
    "run_at timestamp, check_name string, severity string, ok boolean, value string",
)
df.write.mode("append").saveAsTable("data_quality_results")
spark.sql("COMMENT ON TABLE data_quality_results IS 'History of data quality check runs (04_data_quality)'")
display(df)

# COMMAND ----------

failed = [r for r in results if not r.ok and r.severity == "error"]
warned = [r for r in results if not r.ok and r.severity == "warn"]
for r in warned:
    print(f"WARN  {r.name}: {r.value}")
if failed:
    raise AssertionError("data quality failed:\n" + "\n".join(f"  {r.name}: {r.value}" for r in failed))
print(f"{len(results) - len(warned)}/{len(results)} checks passed, {len(warned)} warnings")
