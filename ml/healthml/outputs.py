"""Turn experiment results into the output tables (shared by run_all.py and the Databricks notebooks)."""

from __future__ import annotations

import pandas as pd


def forecast_metrics_table(metrics: dict, target: str) -> pd.DataFrame:
    rows = [
        {"target": target, "model": model, "split": part, **{k: v for k, v in m.items()}}
        for model, parts in metrics.items()
        for part, m in parts.items()
    ]
    return pd.DataFrame(rows)


def output_tables(results: dict) -> dict[str, pd.DataFrame]:
    """Map of output table name (without ``ml_`` prefix) -> DataFrame for whatever experiments ran."""
    t: dict[str, pd.DataFrame] = {}
    if "forecast_resting_hr" in results:
        r = results["forecast_resting_hr"]
        t["forecast_resting_hr"] = r["predictions"][["local_date", "actual", "predicted", "model"]]
        t["forecast_metrics"] = forecast_metrics_table(r["metrics"], "resting_hr")
    if "forecast_hrv_sdnn_ms" in results:
        r = results["forecast_hrv_sdnn_ms"]
        t["forecast_hrv_sdnn_ms"] = r["predictions"][["local_date", "actual", "predicted", "model"]]
        m = forecast_metrics_table(r["metrics"], "hrv_sdnn_ms")
        t["forecast_metrics"] = pd.concat([t.get("forecast_metrics"), m], ignore_index=True)
    if "anomaly" in results:
        r = results["anomaly"]
        t["daily_anomalies"] = r["anomalies"]
        t["similar_days"] = r["similar_days"]
        t["anomaly_episodes"] = r["episodes"]
    if "clustering" in results:
        r = results["clustering"]
        t["day_clusters"] = r["days"]["clusters"]
        t["day_cluster_profiles"] = r["days"]["profiles"]
        t["workout_clusters"] = r["workouts"]["clusters"]
    if "ecg" in results:
        r = results["ecg"]
        t["ecg_beat_anomalies"] = r["beats"][
            [
                "ecg_id",
                "recorded_local",
                "beat_index",
                "r_peak_s",
                "rr_ms",
                "anomaly_score",
                "is_anomaly",
                "split",
                "flag_reason",
                "reconstruction_mse",
            ]
        ]
        t["ecg_recordings"] = r["recordings"]
    return t
