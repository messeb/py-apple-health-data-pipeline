# Getting Started

This page takes you from your iPhone to a running app on your laptop in about 10 minutes.

## 1. Export Your Data from the iPhone

1. Open the **Health** app.
2. Tap your **profile picture** (top right), then **Export All Health Data**.
3. Confirm. Exporting can take a few minutes for years of data.
4. Send the resulting `export.zip` to your computer (AirDrop, Files, …).
5. Unzip it. You get a folder `apple_health_export/` that contains:
   - `export.xml` (the main file, often several GB);
   - `export_cda.xml` (the same data in another format; the pipeline doesn't need it);
   - `electrocardiograms/` (ECG recordings, if you have an Apple Watch with ECG);
   - `workout-routes/` (GPS tracks);
   - `clinical-records/` (only if you connected a health provider).

> The export contains your name, date of birth and detailed health history. Keep it out of git (the
> repository's `.gitignore` already excludes it) and never attach it to an issue. See [Privacy](privacy.md).

## 2. Install

You need [uv](https://docs.astral.sh/uv/) and macOS, Linux or WSL. uv installs Python 3.11 for you, the same version Databricks Apps use.

```bash
git clone https://github.com/messeb/py-apple-health-data-pipeline.git
cd py-apple-health-data-pipeline
make setup              # creates .venv and installs everything (pipeline, app, ML, dev tools)
```

The lighter alternative installs only the pipeline and the app, without PyTorch:

```bash
uv venv --python 3.11 && uv pip install -e ".[app]"
```

## 3. Try It Without Your Own Data (Optional)

```bash
make demo
```

This generates a realistic **synthetic** export for the made-up person "Jane Appleseed". It then runs the
whole pipeline and the ML suite on it, and prints the command to open the app. It's the quickest
way to see what you'll get.

## 4. Run the Pipeline on Your Export

Move or symlink your export into the project folder, then build all layers:

```bash
mv ~/Downloads/apple_health_export .        # or: ln -s ~/Downloads/apple_health_export .
make pipeline                               # bronze → silver → gold, about 1 minute per 3 GB of export.xml
```

What you get:

- `data/bronze|silver|gold/*.parquet`, one file per table;
- `data/health.duckdb`, all tables in one database you can query.

**Timezone.** Apple writes every timestamp with the timezone of the moment you exported, so winter data
would otherwise be one hour off. The pipeline fixes this. By default it detects your timezone from the export
(`--timezone auto`). Set it explicitly if you moved during the recorded period, or if it detects the wrong
zone:

```bash
make pipeline TIMEZONE=America/New_York
# or: .venv/bin/health-pipeline all --export-dir apple_health_export --data-dir data --timezone America/New_York
```

Other options of the CLI:

```bash
.venv/bin/health-pipeline --help
.venv/bin/health-pipeline bronze      # just parse the export
.venv/bin/health-pipeline silver      # rebuild silver from bronze (seconds)
.venv/bin/health-pipeline gold        # rebuild gold from silver (seconds)
```

## 5. Open the App

```bash
make app      # = .venv/bin/streamlit run app/app.py → http://localhost:8501
```

See [Streamlit App](app.md) for what each page shows.

## 6. Run the ML Suite (Optional)

```bash
make ml             # forecasts, anomalies, clusters, ECG beats → data/ml/*.parquet
```

Then reload the app. The **ML insights** page now shows your results. [ML](ml.md) explains the
approaches.

## 7. Query Your Data Directly

```bash
duckdb data/health.duckdb
```

```sql
SELECT year, ROUND(AVG(steps)) AS steps, ROUND(AVG(resting_hr), 1) AS rhr
FROM gold_daily_summary GROUP BY year ORDER BY year;
```

The tables are described in [Data Model](data-model.md).

## Next Steps

- Take it to the cloud: [Databricks Step by Step](databricks.md).
- Something broke: [Troubleshooting](troubleshooting.md).
