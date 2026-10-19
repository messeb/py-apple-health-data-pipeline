# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold: daily, hourly and workout marts
# MAGIC
# MAGIC Same mechanism as silver, for `LAYERS["gold"]`. `gold_daily_summary` (one row per day) is the
# MAGIC main input for the app, the Lakeview dashboard and the ML notebooks.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

from health_lakehouse.spark_runner import build_layer_spark

spark.sql(f"USE {FQ}")  # = USE CATALOG + USE SCHEMA
counts = build_layer_spark(spark, "gold", {"TIMEZONE": TIMEZONE})

# COMMAND ----------

display(spark.createDataFrame(sorted(counts.items()), "table string, row_count long"))
