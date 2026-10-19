"""Optional MLflow logging. Locally (no MLflow installed) every call is a no-op."""

from __future__ import annotations

import contextlib
import os


class Tracker:
    def __init__(self, enabled: bool | None = None, experiment: str | None = None):
        if enabled is None:
            enabled = bool(os.environ.get("DATABRICKS_RUNTIME_VERSION")) or bool(os.environ.get("MLFLOW_TRACKING_URI"))
        self.mlflow = None
        if enabled:
            try:
                import mlflow

                self.mlflow = mlflow
            except Exception as exc:  # MLflow missing: keep going without it
                print(f"[tracking] MLflow disabled: {exc}")
            if self.mlflow is not None and experiment:
                try:
                    self.mlflow.set_experiment(experiment)
                except Exception as exc:  # fall back to the notebook's default experiment
                    print(f"[tracking] set_experiment({experiment!r}) failed, using default: {exc}")

    @contextlib.contextmanager
    def run(self, name: str):
        if self.mlflow is None:
            yield self
            return
        try:
            ctx = self.mlflow.start_run(run_name=name, nested=self.mlflow.active_run() is not None)
        except Exception as exc:
            print(f"[tracking] could not start run: {exc}")
            yield self
            return
        with ctx:
            yield self

    def _safe(self, fn, *args, **kwargs):
        if self.mlflow is None:
            return
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            print(f"[tracking] {fn.__name__} failed: {exc}")

    def params(self, params: dict):
        self._safe(lambda p: self.mlflow.log_params({k: str(v)[:250] for k, v in p.items()}), params)

    def metrics(self, metrics: dict, prefix: str = ""):
        clean = {
            f"{prefix}{k}".replace("@", "_at_"): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and v == v
        }
        self._safe(lambda m: self.mlflow.log_metrics(m), clean)

    def model(self, model, name: str):
        def log(m):
            try:
                self.mlflow.pytorch.log_model(m.cpu(), name=name)  # MLflow >= 3
            except TypeError:
                self.mlflow.pytorch.log_model(m.cpu(), artifact_path=name)

        self._safe(log, model)

    def figure(self, path):
        self._safe(lambda p: self.mlflow.log_artifact(str(p), artifact_path="figures"), path)
