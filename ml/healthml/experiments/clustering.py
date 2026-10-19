"""(e) Unsupervised "day types" and "workout types".

PCA (torch SVD) for the 2D map, k-means (torch, k-means++), plus sklearn Ward
hierarchical clustering and DBSCAN as comparisons. k is chosen by silhouette score.
Clusters are named from their z-score profiles so the app can show readable labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from ..config import FIG_DIR, SEED, first_watch_day
from ..data import load_daily
from ..models import kmeans, pca
from ..plotting import GRAY, SERIES, plt, save
from ..tracking import Tracker

# Behaviour only: physiology (RHR, HRV) drifts with fitness over years and would turn the
# clusters into "eras". It is reported per cluster instead.
DAY_FEATURES = [
    "log_steps",
    "walk_run_km",
    "cycling_km",
    "flights_climbed",
    "active_kcal",
    "exercise_min",
    "stand_min",
    "workout_min",
]
PROFILE_EXTRA = ["steps", "hr_mean", "resting_hr", "hrv_sdnn_ms", "is_weekend"]
PHRASES = {
    "cycling_km": "cycling",
    "walk_run_km": "walking",
    "log_steps": "high-step",
    "flights_climbed": "stairs",
    "workout_min": "long workout",
    "exercise_min": "exercise",
    "active_kcal": "high-burn",
    "resting_hr": "elevated RHR",
    "hrv_sdnn_ms": "high HRV",
    "hr_mean": "high HR",
    "stand_min": "on-feet",
}


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)


def _choose_k(z: np.ndarray, ks=range(3, 8)) -> tuple[int, dict]:
    t = torch.as_tensor(z, dtype=torch.float32)
    scores = {}
    sample = np.random.default_rng(SEED).choice(len(z), size=min(len(z), 2500), replace=False)
    for k in ks:
        _, labels, _ = kmeans(t, k, n_init=5, seed=SEED)
        scores[k] = float(silhouette_score(z[sample], labels.numpy()[sample]))
    return max(scores, key=scores.get), scores


def _name_clusters(profile_z: pd.DataFrame) -> dict[int, str]:
    names = {}
    activity = ["log_steps", "active_kcal", "exercise_min", "workout_min", "cycling_km", "walk_run_km"]
    for c, row in profile_z.iterrows():
        if row[activity].max() < 0 and row[activity].mean() < -0.4:
            name = "rest / low-activity day"
        else:
            top = row.sort_values(ascending=False)
            picks = [PHRASES[f] for f, v in top.items() if v > 0.5][:2]
            name = (" + ".join(picks) + " day") if picks else "average day"
        names[c] = name
    seen: dict[str, int] = {}
    for c in sorted(names):
        n = names[c]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} ({seen[n]})"
    return names


def cluster_days(daily: pd.DataFrame, start: str | None = None) -> dict:
    # Days from when the watch was reliably worn (coverage-based); phone-only days lack kcal/HR.
    d = daily.loc[start or first_watch_day(daily) :].copy()
    d["log_steps"] = np.log1p(d["steps"])
    d = d[d["steps"].notna() & d["active_kcal"].notna()]
    x = d[DAY_FEATURES].copy()
    x = x.fillna(x.median())
    z = _zscore(x.to_numpy(float))
    k, sil = _choose_k(z)
    t = torch.as_tensor(z, dtype=torch.float32)
    _, labels, inertia = kmeans(t, k, n_init=10, seed=SEED)
    labels = labels.numpy()
    scores, _, evr = pca(t, 2)
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(z)
    nn = NearestNeighbors(n_neighbors=10).fit(z)
    eps = float(np.quantile(nn.kneighbors(z)[0][:, -1], 0.9))
    db = DBSCAN(eps=eps, min_samples=10).fit_predict(z)

    prof_z = pd.DataFrame(z, columns=DAY_FEATURES).groupby(labels).mean()
    names = _name_clusters(prof_z)
    raw = d[DAY_FEATURES + PROFILE_EXTRA].groupby(labels).mean()
    raw.insert(0, "n_days", pd.Series(labels).value_counts().sort_index())
    raw.insert(0, "cluster_label", [names[c] for c in raw.index])
    raw.index.name = "cluster"
    out = pd.DataFrame(
        {
            "local_date": d.index.date,
            "cluster": labels.astype(int),
            "x": scores[:, 0].numpy(),
            "y": scores[:, 1].numpy(),
            "cluster_label": [names[c] for c in labels],
        }
    )
    by_year = pd.crosstab(d.index.year, labels, normalize="index")
    metrics = {
        "k": k,
        "silhouette_by_k": sil,
        "inertia": inertia,
        "pca_explained_2d": float(evr.sum()),
        "ari_kmeans_vs_ward": float(adjusted_rand_score(labels, ward)),
        "dbscan_eps": eps,
        "dbscan_n_clusters": len(set(db) - {-1}),
        "dbscan_noise_share": float((db == -1).mean()),
        "ari_kmeans_vs_dbscan": float(adjusted_rand_score(labels, db)),
    }
    return {
        "clusters": out,
        "profiles": raw.reset_index(),
        "by_year": by_year,
        "names": names,
        "metrics": metrics,
        "pca_components": pd.DataFrame(_pca_loadings(t), index=DAY_FEATURES, columns=["pc1", "pc2"]),
    }


def _pca_loadings(t):
    _, comps, _ = pca(t, 2)
    return comps.T.numpy()


def cluster_workouts(source) -> dict:
    w = source.load("gold_workouts")
    w["local_date"] = pd.to_datetime(w["local_date"])
    w = w[w["duration_min"].between(3, 600)].copy()
    feats = pd.DataFrame(
        {
            "log_duration": np.log(w["duration_min"]),
            "kcal_per_min": w["kcal_per_min"].clip(0, 25),
            "hr_avg": w["hr_avg"],
            "has_hr": w["hr_avg"].notna().astype(float),
            "speed_kmh": w["avg_speed_kmh"].clip(0, 60).fillna(0),
            "has_distance": w["distance_km"].fillna(0).gt(0).astype(float),
            "start_hour": w["start_hour"],
            "is_indoor": w["is_indoor"].map({True: 1.0, False: 0.0, "true": 1.0, "false": 0.0}).fillna(0.5),
        },
        index=w.index,
    )
    feats = feats.fillna(feats.median())
    z = _zscore(feats.to_numpy(float))
    k, sil = _choose_k(z, range(3, 9))
    t = torch.as_tensor(z, dtype=torch.float32)
    _, labels, _ = kmeans(t, k, n_init=10, seed=SEED)
    labels = labels.numpy()
    scores, _, _ = pca(t, 2)
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(z)
    nn = NearestNeighbors(n_neighbors=8).fit(z)
    eps = float(np.quantile(nn.kneighbors(z)[0][:, -1], 0.9))
    db = DBSCAN(eps=eps, min_samples=8).fit_predict(z)
    act = w["activity"].to_numpy()
    # Profile on raw values (hr_avg unimputed) so medians are not pulled to the fill value.
    profile = (
        feats.assign(cluster=labels, duration_min=w["duration_min"], activity=act, hr_avg=w["hr_avg"])
        .groupby("cluster")
        .agg(
            n=("activity", "size"),
            top_activity=("activity", lambda s: s.value_counts().index[0]),
            top_activity_share=("activity", lambda s: s.value_counts(normalize=True).iloc[0]),
            duration_min=("duration_min", "median"),
            kcal_per_min=("kcal_per_min", "median"),
            hr_avg=("hr_avg", "median"),
            speed_kmh=("speed_kmh", "median"),
            start_hour=("start_hour", "median"),
        )
    )
    out = pd.DataFrame(
        {
            "workout_id": w["workout_id"].to_numpy(),
            "local_date": w["local_date"].dt.date,
            "activity": act,
            "cluster": labels.astype(int),
            "x": scores[:, 0].numpy(),
            "y": scores[:, 1].numpy(),
        }
    )
    metrics = {
        "k": k,
        "silhouette_by_k": sil,
        "n_workouts": len(w),
        "ari_vs_activity_kmeans": float(adjusted_rand_score(act, labels)),
        "nmi_vs_activity_kmeans": float(normalized_mutual_info_score(act, labels)),
        "ari_vs_activity_ward": float(adjusted_rand_score(act, ward)),
        "ari_vs_activity_dbscan": float(adjusted_rand_score(act, db)),
        "dbscan_n_clusters": len(set(db) - {-1}),
        "dbscan_noise_share": float((db == -1).mean()),
    }
    return {"clusters": out, "profiles": profile.reset_index(), "metrics": metrics}


def run(source, fig_dir=FIG_DIR, tracker=None) -> dict:
    tracker = tracker or Tracker(enabled=False)
    daily = load_daily(source)
    days = cluster_days(daily)
    workouts = cluster_workouts(source)
    figs = _figures(days, workouts, fig_dir)
    with tracker.run("clustering"):
        tracker.params({"day_k": days["metrics"]["k"], "workout_k": workouts["metrics"]["k"]})
        tracker.metrics({k: v for k, v in days["metrics"].items() if not isinstance(v, dict)}, prefix="days.")
        tracker.metrics({k: v for k, v in workouts["metrics"].items() if not isinstance(v, dict)}, prefix="workouts.")
        for f in figs:
            tracker.figure(f)
    return {"days": days, "workouts": workouts, "figures": figs}


def _figures(days, workouts, fig_dir):
    c = days["clusters"]
    names = days["names"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), width_ratios=[1.1, 1])
    ax = axes[0]
    for i in sorted(names):
        m = c["cluster"] == i
        ax.scatter(
            c.loc[m, "x"], c.loc[m, "y"], s=5, alpha=0.5, color=SERIES[i % 8], label=f"{i}: {names[i]} ({m.sum()})"
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Day types (k-means on standardized daily vectors, PCA map)")
    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax = axes[1]
    by = days["by_year"]
    bottom = np.zeros(len(by))
    for i in by.columns:
        ax.bar(
            by.index.astype(str), by[i], bottom=bottom, color=SERIES[i % 8], edgecolor="#fcfcfb", lw=1, label=names[i]
        )
        bottom += by[i].to_numpy()
    ax.set_title("Share of day types per year")
    ax.set_ylabel("share of days")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    f1 = save(fig, fig_dir / "day_clusters.png")

    wc = workouts["clusters"]
    fig, ax = plt.subplots(figsize=(7, 5))
    top = wc["activity"].value_counts().index[:6]
    for i, a in enumerate(top):
        m = wc["activity"] == a
        ax.scatter(wc.loc[m, "x"], wc.loc[m, "y"], s=10, alpha=0.6, color=SERIES[i], label=a)
    m = ~wc["activity"].isin(top)
    ax.scatter(wc.loc[m, "x"], wc.loc[m, "y"], s=10, alpha=0.6, color=GRAY, label="other")
    ax.set_title(
        f"Workouts in PCA space (k-means k={workouts['metrics']['k']}, "
        f"ARI vs activity {workouts['metrics']['ari_vs_activity_kmeans']:.2f})"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    f2 = save(fig, fig_dir / "workout_clusters.png")
    return [f1, f2]
