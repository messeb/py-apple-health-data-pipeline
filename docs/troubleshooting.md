# Troubleshooting

## Local

| Symptom | Cause and fix |
|---|---|
| `missing bronze file …` | Run `make pipeline` (or `health-pipeline bronze`) before `silver`/`gold`. |
| `export.xml` not found | The export must be at `./apple_health_export/export.xml`, or pass `EXPORT_DIR=/path/to/apple_health_export`. |
| Daily totals look shifted by one hour, or days split at the wrong time | The timezone was detected wrongly, or you moved. Rebuild with `make pipeline TIMEZONE=<IANA zone>`, e.g. `Europe/London`. |
| Steps are higher than in the Health app | Expected in rare hours where third-party apps write overlapping steps. The pipeline keeps one source per hour; the Health app uses a finer-grained priority list. |
| Sleep pages are almost empty | You probably didn't wear the watch at night. The Sleep page shows its coverage first. |
| App says "no data" | Build the pipeline first. For another folder, set `HEALTH_DATA_DIR=/path/to/data`. |
| App theme looks wrong after code edits | Restart Streamlit. It doesn't hot-reload `charts.py`, `data.py` or `ui.py`. |
| `ml/run_all.py` skips experiments | Not enough data for that experiment (see [ML → How Much Data You Need](ml.md#how-much-data-you-need)). |
| Spark parity tests are skipped | They need Java 17 or 21 (`JAVA_HOME`, or a JDK in `.cache/jdk/`) and `uv pip install -e ".[spark]"`. |
| `uv` installs a CUDA build of torch on Linux | Use the CPU index: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`. |

## Databricks

| Symptom | Cause and fix |
|---|---|
| `Catalog 'main' does not exist` | New and Free Edition workspaces call it `workspace`, the default here. Check with `databricks catalogs list`, and see [Databricks Step 1](databricks.md#step-1-check-the-workspace). |
| `validate`: warehouse not found | Your warehouse isn't called "Serverless Starter Warehouse". Deploy with `--var warehouse_id=<id>` ([Step 5](databricks.md#step-5-validate-and-deploy-the-bundle)). |
| `deploy`: permission error on the warehouse | Granting the app `CAN_USE` needs `CAN_MANAGE` on the warehouse. Ask an admin, or use a warehouse you created yourself. |
| `PERMISSION_DENIED … CREATE SCHEMA / CREATE TABLE` | You lack rights in the catalog. Use one you own ([Step 1](databricks.md#step-1-check-the-workspace)). |
| Setup task: "no export found" | Upload to `/Volumes/<catalog>/<schema>/raw/apple_health_export/export.xml` or `/Volumes/<catalog>/<schema>/raw/export.zip` ([Step 7](databricks.md#step-7-upload-the-apple-health-export)). |
| ML task fails installing packages | Serverless needs internet access to PyPI for the `%pip` cells. Ask your admin. |
| App build: `No matching distribution found for numpy==…` | The Apps runtime is Python 3.11. `app/requirements.txt` uses lower bounds only; don't pin versions that need a newer Python. |
| App: permission denied / table not found | Grant the app's service principal `USE CATALOG`, `USE SCHEMA`, `SELECT` ([Step 11](databricks.md#step-11-give-the-app-read-access-to-your-tables)), and check `HEALTH_CATALOG` / `HEALTH_SCHEMA` in `app/app.yaml`. |
| App shows wrong hours | The tables were built with another timezone. Rerun the job with `--params timezone=<your zone>`. |

Still stuck? Open an issue with the error message and your environment. **Don't attach your export or any
real data**; reproduce with `make demo` if you can.
