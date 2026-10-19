# Databricks notebook source
# MAGIC %md
# MAGIC # Classification, workout regression and descriptive insights
# MAGIC
# MAGIC Next-day ring/workout classifiers (logistic regression vs. classic classifiers), workout kcal / HR
# MAGIC regression and the descriptive insights. Logs metrics and figures to MLflow; writes no tables (the
# MAGIC table-producing experiments are notebooks 10-13).
# MAGIC
# MAGIC Uses the `ml/healthml` package from this repo. Parameters: `catalog` (default `main`), `schema`
# MAGIC (default `health`), `mlflow_experiment`. Runs on serverless or CPU/GPU clusters; the first cell
# MAGIC pip-installs torch, scikit-learn, scipy and matplotlib (a no-op on ML runtimes; MLflow is preinstalled).

# COMMAND ----------

# MAGIC %pip install -q "torch>=2.4" "scikit-learn>=1.5" "scipy>=1.11" "matplotlib>=3.9" "pandas>=2.2" "pyarrow>=16"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "health", "Schema")
dbutils.widgets.text("mlflow_experiment", "/Shared/health_ml", "MLflow experiment")
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
from healthml.sources import SparkSource
from healthml.tracking import Tracker

source = SparkSource(spark, catalog, schema)
tracker = Tracker(enabled=True, experiment=mlflow_experiment)  # no-op if MLflow is unavailable
FIG_DIR = Path(tempfile.mkdtemp(prefix="healthml_figures_"))
ensure_dirs(FIG_DIR)
print(f"repo={REPO_ROOT} source={catalog}.{schema} figures={FIG_DIR}")

# COMMAND ----------

from healthml.experiments import classify, insights, workouts

cls = classify.run(source, fig_dir=FIG_DIR, tracker=tracker)
for label, r in cls["results"].items():
    print(label, {m: round(v["test"]["auc"], 3) for m, v in r["metrics"].items()})
wk = workouts.run(source, fig_dir=FIG_DIR, tracker=tracker)
for target, r in wk["results"].items():
    print(target, {m: round(v["test"]["mae"], 2) for m, v in r["metrics"].items()})
ins = insights.run(source, fig_dir=FIG_DIR)
display(ins["yearly"].reset_index())
