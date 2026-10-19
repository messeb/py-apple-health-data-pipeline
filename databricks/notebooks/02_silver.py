# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: typed, deduplicated tables
# MAGIC
# MAGIC Runs every file of `src/health_lakehouse/sql/silver/` in manifest order (`LAYERS["silver"]`) as
# MAGIC `CREATE OR REPLACE TABLE <name> AS <sql>`. These are the same SQL files the local DuckDB
# MAGIC pipeline runs. The table comment comes from the file's leading `--` lines.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

dbutils.widgets.dropdown("optimize", "true", ["true", "false"], "OPTIMIZE big tables")

from health_lakehouse.spark_runner import build_layer_spark

spark.sql(f"USE {FQ}")  # = USE CATALOG + USE SCHEMA
counts = build_layer_spark(spark, "silver", {"TIMEZONE": TIMEZONE})

# COMMAND ----------

if dbutils.widgets.get("optimize") == "true":
    # Nearly every downstream query filters or groups silver_records by local_date.
    spark.sql("OPTIMIZE silver_records ZORDER BY (local_date, metric)")
    spark.sql("OPTIMIZE silver_route_points")

display(spark.createDataFrame(sorted(counts.items()), "table string, row_count long"))
