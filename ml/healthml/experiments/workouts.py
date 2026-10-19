"""(g) Workout-level regression: active kcal and average HR from "planned" workout attributes.

Inputs are things known before or independent of effort physiology: activity, duration,
distance/speed, elevation, weather, start hour, indoor flag, and calendar time. Heart rate,
METs and kcal are excluded as inputs because Apple derives kcal from HR/METs (that would
be target leakage). Split is chronological (70/15/15 by start time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ..config import FIG_DIR, SEED, NotEnoughData, get_device, set_seed
from ..data import Standardizer
from ..evaluation import regression_metrics
from ..models import MLP
from ..plotting import SERIES, plt, save
from ..tracking import Tracker
from ..train import fit, huber_loss, predict, to_tensors

N_TOP_ACTIVITIES = 6  # the most frequent activities get their own indicator, the rest is "Other"
TARGETS = {"active_kcal": "active kcal", "hr_avg": "average HR (bpm)"}


def build_workout_frame(source) -> pd.DataFrame:
    w = source.load("gold_workouts")
    w["start_local"] = pd.to_datetime(w["start_local"])
    w = w[w["duration_min"].between(3, 600)].sort_values("start_local").reset_index(drop=True)
    if len(w) < 60:
        raise NotEnoughData(f"workouts: only {len(w)} workouts (need 60)")
    top = w["activity"].value_counts().index[:N_TOP_ACTIVITIES].tolist()
    act = w["activity"].where(w["activity"].isin(top), "Other")
    x = pd.DataFrame(
        {
            "duration_min": w["duration_min"],
            "distance_km": w["distance_km"].fillna(0),
            "speed_kmh": w["avg_speed_kmh"].clip(0, 60).fillna(0),
            "elevation_gain_m": w["elevation_gain_m"].fillna(0),
            "weather_temp_c": w["weather_temp_c"],
            "weather_humidity_pct": w["weather_humidity_pct"],
            "start_hour": w["start_hour"],
            "is_indoor": w["is_indoor"].map({True: 1.0, False: 0.0, "true": 1.0, "false": 0.0}),
            "years_since_first": (w["start_local"] - w["start_local"].min()).dt.days / 365.25,
        }
    )
    for a in [*top, "Other"]:
        x[f"act_{a}"] = (act == a).astype(float)
    x["activity"] = act
    x["start_local"] = w["start_local"]
    x["active_kcal"] = w["active_kcal"]
    x["hr_avg"] = w["hr_avg"]
    return x


def _run_target(df: pd.DataFrame, target: str, tracker: Tracker) -> dict:
    d = df[df[target].notna() & (df[target] > 0)].reset_index(drop=True)
    n = len(d)
    parts = np.array(["train"] * n, dtype=object)
    parts[int(0.70 * n) :] = "val"
    parts[int(0.85 * n) :] = "test"
    feat = [c for c in d.columns if c not in ("activity", "start_local", "active_kcal", "hr_avg")]
    x_raw = d[feat].to_numpy(float)
    y = d[target].to_numpy(float)
    tr, va, te = (parts == p for p in ("train", "val", "test"))
    scaler = Standardizer().fit(x_raw[tr])
    x = scaler.transform(x_raw)
    x_imp = np.where(np.isnan(x_raw), scaler.mean_, x_raw)

    preds = {}
    # Baseline: per-activity median of the train set (kcal per minute x duration for kcal).
    if target == "active_kcal":
        rate = (d[target] / d["duration_min"])[tr].groupby(d["activity"][tr]).median()
        preds["activity_median_rate_baseline"] = (
            d["activity"].map(rate).fillna(rate.median()).to_numpy() * d["duration_min"].to_numpy()
        )
    else:
        med = d[target][tr].groupby(d["activity"][tr]).median()
        preds["activity_median_baseline"] = d["activity"].map(med).fillna(d[target][tr].median()).to_numpy()
    models = {
        "linear_ridge": Ridge(1.0),
        "decision_tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=SEED),
        "random_forest": RandomForestRegressor(400, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, min_samples_leaf=10, random_state=SEED
        ),
        "knn_10": KNeighborsRegressor(10, weights="distance"),
        "svr_rbf": SVR(C=float(np.std(y[tr])) * 3, epsilon=float(np.std(y[tr])) * 0.05),
    }
    for name, m in models.items():
        xin = x if name in ("linear_ridge", "knn_10", "svr_rbf") else x_imp
        m.fit(xin[tr], y[tr])
        preds[name] = m.predict(xin)

    y_mu, y_sd = y[tr].mean(), y[tr].std()
    device = get_device()
    outs = []
    for s in range(3):
        set_seed(SEED + s)
        xt, yt = to_tensors(x, (y - y_mu) / y_sd, device=device)
        mlp = MLP(x.shape[1], (64, 32), 0.2).to(device)
        fit(
            mlp,
            huber_loss,
            (xt[tr], yt[tr]),
            (xt[va], yt[va]),
            epochs=600,
            lr=2e-3,
            weight_decay=1e-3,
            batch_size=64,
            patience=40,
            seed=SEED + s,
        )
        outs.append(predict(mlp, xt) * y_sd + y_mu)
    preds["mlp_torch"] = np.mean(outs, axis=0)

    metrics = {
        name: {p: regression_metrics(y[m], pr[m]) for p, m in (("val", va), ("test", te))} for name, pr in preds.items()
    }
    rf = models["random_forest"]
    pi = permutation_importance(
        rf, x_imp[te], y[te], n_repeats=10, random_state=SEED, scoring="neg_mean_absolute_error"
    )
    imp = pd.DataFrame({"feature": feat, "mae_increase": pi.importances_mean}).sort_values(
        "mae_increase", ascending=False
    )
    with tracker.run(f"workouts_{target}"):
        for name, m in metrics.items():
            tracker.metrics({k: v for k, v in m["test"].items() if k != "n"}, prefix=f"{name}.test.")
    return {
        "metrics": metrics,
        "importance": imp,
        "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "test_period_start": str(d["start_local"][te].min().date()),
    }


def run(source, fig_dir=FIG_DIR, tracker=None) -> dict:
    tracker = tracker or Tracker(enabled=False)
    df = build_workout_frame(source)
    res = {t: _run_target(df, t, tracker) for t in TARGETS}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (t, r) in zip(axes, res.items(), strict=False):
        imp = r["importance"].head(8)[::-1]
        ax.barh(imp["feature"], imp["mae_increase"], color=SERIES[0], height=0.6)
        ax.set_title(f"{TARGETS[t]}: permutation importance (RF, test)")
        ax.set_xlabel("MAE increase when shuffled")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return {"results": res, "figures": [save(fig, fig_dir / "workout_importance.png")]}
