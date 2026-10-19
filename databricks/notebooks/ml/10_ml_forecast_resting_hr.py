# Databricks notebook source
# MAGIC %md
# MAGIC # Next-day resting HR / HRV forecasting
# MAGIC
# MAGIC Reads `gold_daily_summary`, compares naive baselines (persistence, 7-day mean, seasonal naive) with
# MAGIC linear regression, MLP, GRU, LSTM and a small Transformer (PyTorch), histogram gradient boosting and
# MAGIC optionally zero-shot `amazon/chronos-bolt-small`. Time split relative to the data: the last 20%
# MAGIC of days is test, the 15% before it validation, the rest train.
# MAGIC
# MAGIC Writes `ml_forecast_resting_hr` (local_date, actual, predicted, model), `ml_forecast_hrv_sdnn_ms`
# MAGIC and `ml_forecast_metrics`.
# MAGIC
# MAGIC Uses the `ml/healthml` package from this repo. Parameters: `catalog` (default `main`), `schema`
# MAGIC (default `health`), `mlflow_experiment`. Runs on serverless or CPU/GPU clusters; the first cell
# MAGIC pip-installs torch, scikit-learn, scipy and matplotlib (a no-op on ML runtimes; MLflow is preinstalled).

# COMMAND ----------

# MAGIC %pip install -q "torch>=2.4" "scikit-learn>=1.5" "scipy>=1.11" "matplotlib>=3.9" "pandas>=2.2" "pyarrow>=16"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md Optional: add `chronos-forecasting` to the `%pip install` cell above and set `use_chronos=true`.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "health", "Schema")
dbutils.widgets.text("mlflow_experiment", "/Shared/health_ml", "MLflow experiment")
dbutils.widgets.dropdown(
    "use_chronos", "false", ["true", "false"], "Zero-shot Chronos (needs chronos-forecasting + HF download)"
)
dbutils.widgets.text("n_seeds", "3", "Seeds per neural model")
use_chronos = dbutils.widgets.get("use_chronos") == "true"
n_seeds = int(dbutils.widgets.get("n_seeds"))
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
mlflow_experiment = dbutils.widgets.get("mlflow_experiment")

# COMMAND ----------

# Make ml/healthml importable: the repo is synced as workspace files, and on DBR 14+ the
# working directory of a notebook is its own folder (<repo>/databricks/notebooks/ml).
import os
import sys
import tempfile
from pathlib import Path


def _find_repo_root():
    starts = [Path.cwd()]
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        starts.append(Path("/Workspace" + nb if not nb.startswith("/Workspace") else nb).parent)
    except Exception:
        pass
    for start in starts:
        for cand in [start, *start.parents]:
            if (cand / "ml" / "healthml").is_dir():
                return cand
    raise FileNotFoundError("ml/healthml not found; is the repo synced as workspace files?")


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "ml"))
os.environ.setdefault(
    "HF_HOME", "/local_disk0/hf_cache" if os.access("/local_disk0", os.W_OK) else tempfile.gettempdir() + "/hf_cache"
)
os.environ.setdefault("HEALTHML_DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")

from healthml.config import ensure_dirs
from healthml.databricks_utils import write_tables
from healthml.outputs import output_tables
from healthml.sources import SparkSource
from healthml.tracking import Tracker

source = SparkSource(spark, catalog, schema)
tracker = Tracker(enabled=True, experiment=mlflow_experiment)  # no-op if MLflow is unavailable
FIG_DIR = Path(tempfile.mkdtemp(prefix="healthml_figures_"))
ensure_dirs(FIG_DIR)
print(f"repo={REPO_ROOT} source={catalog}.{schema} figures={FIG_DIR}")

# COMMAND ----------

from healthml.experiments import forecast

results = {}
for target in ("resting_hr", "hrv_sdnn_ms"):
    results[f"forecast_{target}"] = forecast.run(
        source, target=target, n_seeds=n_seeds, use_chronos=use_chronos, fig_dir=FIG_DIR, tracker=tracker
    )
    m = results[f"forecast_{target}"]["metrics"]
    print(target, {k: round(v["test"]["mae"], 3) for k, v in m.items() if "test" in v})

# COMMAND ----------

tables = output_tables(results)
display(tables["forecast_metrics"])
write_tables(spark, tables, catalog, schema)
