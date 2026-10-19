# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0, minor versions may
change table schemas. Breaking changes are called out under **Changed**.

## [Unreleased]

## [0.1.0] - 2026-10-19

Initial public release of Apple Health Data Pipeline.

### Added

- **Bronze ingestion** (`health-lakehouse bronze`): a streaming, memory-flat parser for `export.xml`
  (records, correlations, workouts with events, statistics and routes, activity summaries, `<Me>`),
  ECG CSVs (locale-aware, name and date of birth dropped), GPX workout routes and FHIR clinical
  records. Output is raw-string Parquet.
- **Silver and gold layers** as plain Databricks SQL files under `src/health_lakehouse/sql/`: typed,
  deduplicated records with correct UTC and local time across DST, sleep segments, workouts,
  HRV beats, ECG samples, route points, plus daily, hourly and weekly gold tables for activity,
  vitals, sleep, workouts and ECG.
- **Two engines, one SQL codebase:** a local DuckDB runner (through a small `sql_compat` translation
  layer) and the Databricks notebooks run the same files.
- **Lakeview dashboard** (`databricks/dashboards/health.lvdash.json`) with five pages, deployed by the
  bundle and pointed at the configured catalog and schema.
- **Timezone auto-detection** (`--timezone auto`) from the export's `HKTimeZone` metadata.
- **Data quality checks** (row counts, uniqueness, plausibility ranges) that fail the Databricks job
  on errors.
- **Databricks Asset Bundle** (`databricks.yml`): a serverless job (setup, bronze, silver, gold, data
  quality, then five parallel ML notebooks with MLflow tracking) and the Streamlit app as a Databricks
  App on a SQL warehouse. No workspace host is hard-coded.
- **Streamlit app** with 8 pages: overview, activity, heart, workouts with route maps, body
  and gait, sleep, ECG viewer and ML insights. It runs locally on Parquet or on Databricks.
- **PyTorch ML suite** (`ml/run_all.py`): resting HR / HRV forecasting against baselines (optional
  zero-shot Chronos-Bolt), daily anomaly detection, day and workout clustering, ECG beat anomalies,
  workout classification and insights.
- **Synthetic sample data** (`health-lakehouse sample`, `make demo`): a realistic, fictional export
  for demos, tests and bug reports.
- **Privacy tooling:** a hardened `.gitignore` and `scripts/check_no_health_data.py`, a pre-commit
  and CI guard against committing exports, data files, workspace hosts and tokens.
- Project tooling: `Makefile`, pre-commit (ruff, the privacy guard and standard hygiene hooks),
  GitHub Actions CI, Dependabot, issue and PR templates, dev container, contributing guide, code of
  conduct and security policy.

[Unreleased]: https://github.com/messeb/py-apple-health-data-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/messeb/py-apple-health-data-pipeline/releases/tag/v0.1.0
