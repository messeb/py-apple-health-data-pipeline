# Contributing to Apple Health Data Pipeline

Thanks for helping. This project turns a personal Apple Health export into a lakehouse, so privacy
comes first in every contribution.

> **Never attach a real Apple Health export, or anything derived from one, to an issue, a pull
> request, a discussion or a test fixture.** That covers `export.xml`, ECG CSVs, GPX routes,
> Parquet/DuckDB files, screenshots of your app and ML reports run on your own data. To
> reproduce a problem, use the synthetic generator (`make sample`) and describe what differs in
> your export, such as the iOS version, export language, device or metric name.

## Development Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.11 (the Databricks Apps runtime). Java 17+ is only needed for the
Spark tests.

```bash
make setup                      # uv venv + all extras (app, ml, dev, spark)
.venv/bin/pre-commit install    # ruff, YAML/whitespace checks and the health-data guard on every commit
make demo                       # synthetic export -> pipeline -> quick ML run, no personal data needed
```

Without `make`:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[app,ml,dev,spark]"
```

Run `make help` to list every target. The CLI is available as both `health-pipeline` and `health-lakehouse`.

Background reading: [Architecture](docs/architecture.md), [Data Model](docs/data-model.md),
[Databricks Guide](docs/databricks.md),
[Streamlit App](docs/app.md), [ML Approaches](docs/ml.md) and [Privacy](docs/privacy.md).

## Tests and Checks

| Command | What it runs |
| --- | --- |
| `make test-fast` | The small unit tests, exactly what CI runs (core deps only, ~10 s). |
| `make test` | The full suite, including app and ML tests. It needs no personal data, and tests that need a real export skip themselves. |
| `make test-spark` | Optional, local only: runs every silver/gold SQL file on local Spark + Delta (`tests/test_spark_sql.py`). Needs Java and the `spark` extra. |
| `make lint` | `ruff check` and `ruff format --check` |
| `make format` | `ruff format` and `ruff check --fix` |
| `make bundle-validate` | `databricks bundle validate` (needs a logged-in Databricks CLI) |
| `.venv/bin/pre-commit run --all-files` | All pre-commit hooks, including `scripts/check_no_health_data.py` |

CI runs lint, the test suite and the privacy guard on every pull request. Run them locally before
you push.

## The SQL Files Are the Single Source of Truth

Silver and gold tables are defined only once, in `src/health_lakehouse/sql/{silver,gold}/*.sql`. The
same files run locally on DuckDB (through the small translation layer in `sql_compat.py`) and on
Databricks (`databricks/notebooks/02_silver.py` and `03_gold.py`). Therefore:

- Write **Databricks SQL**. Don't use DuckDB-only syntax. If DuckDB needs a different spelling, add
  a rewrite to `sql_compat.py` with a test in `tests/test_sql_compat.py`.
- Every change must pass the **Spark parity test** (`make test-spark`). A table that builds on DuckDB
  but fails or differs on Spark is a bug.
- Order matters: `LAYERS` in `src/health_lakehouse/pipeline.py` lists the tables, and a table may
  only read tables listed before it.
- The leading `--` comment lines of a SQL file become the table comment on Databricks, so keep them
  accurate.
- Don't duplicate SQL in notebooks, the app or ML code. Read the gold tables instead.

## Adding a New HealthKit Metric End to End

Example: you want `HKQuantityTypeIdentifierTimeInDaylight` in the app.

1. **Bronze: nothing to do.** The bronze parser stores every `<Record>` generically, whatever its
   type, so the new metric is already in `bronze_records`.
2. **Silver:** `silver_records` derives `metric` from the HealthKit type and types the value.
   Check the unit and value handling for the new metric, such as categories versus quantities or
   unit conversions. Most quantity metrics need no change.
3. **Gold:** add the aggregate to the right gold table, for example
   `AVG(CASE WHEN metric = 'TimeInDaylight' THEN value_num END) AS daylight_min` in
   `gold_daily_vitals.sql`. Choose the aggregation carefully. Additive metrics (steps, distance,
   energy) belong in the metric list of `gold_hourly_activity.sql`, where the source with the
   largest hourly total wins so that iPhone and Watch are not double counted. Discrete metrics
   (heart rate, SpO₂) are averaged. If a downstream summary table such as `gold_daily_summary`
   selects columns explicitly, carry the column through there as well.
4. **Data quality:** add a plausibility check to `src/health_lakehouse/quality.py` if the metric
   has a physical range, and allow a tiny share of outliers, because real exports contain them.
5. **Synthetic data:** teach the sample generator to emit the metric, so the tests and the demo
   cover it.
6. **Streamlit app** (see [docs/app.md](docs/app.md)): add the query to `app/data.py`, a `*_sql` builder that is unit-tested
   without data, and the chart or KPI to the matching page in `app/views/`. Handle the empty
   case, because many users won't have the metric.
7. Document new columns in [docs/data-model.md](docs/data-model.md). Run `make test test-spark lint`, and mention the metric in `CHANGELOG.md` under *Unreleased*.

## Commit Style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(gold): add daily time in daylight
fix(bronze): parse ECG files exported with a German locale
docs: explain the Databricks volume upload
test(sql): cover DST switch in silver_records
chore(deps): bump duckdb to 1.2
```

Keep the subject at 72 characters or fewer, in the imperative mood. Use the body to explain *why*.
Keep each pull request to one logical change.

## Pull Requests

- Fill in the pull request template, including the privacy checkbox.
- Add or update tests. Base new tests on synthetic data, built in the test or generated with the
  sample generator. They must never depend on a real export.
- Update the docs and `CHANGELOG.md` when behaviour changes.
- For changes to the app, use screenshots made from the **synthetic** demo data only.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). See [SECURITY.md](SECURITY.md)
for how to report vulnerabilities privately.
