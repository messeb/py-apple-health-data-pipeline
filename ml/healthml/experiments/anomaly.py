"""(c) Autoencoder anomaly detection on daily physiological vectors, (f) KNN "similar days".

Each day is a vector of absolute values (resting HR, HRV, HR mean/min, walking HR, SpO2,
respiratory rate, steps, active kcal, exercise minutes) plus deviations from a trailing
60-day personal median for resting HR, HRV and HR mean, so slow fitness drift over years
is not flagged but sudden departures are. Missing entries are masked out of the loss.

Anomaly score = mean over observed features of the squared reconstruction error divided by
that feature's typical training error. Days above the 98th percentile are flagged. A PCA
reconstruction with the same latent size is the linear baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..config import FIG_DIR, SEED, NotEnoughData, first_watch_day, get_device, set_seed
from ..data import load_daily
from ..models import MaskedAutoencoder, pca
from ..plotting import GRAY, SERIES, plt, save
from ..tracking import Tracker
from ..train import fit, to_tensors

ABS = [
    "resting_hr",
    "hrv_sdnn_ms",
    "hr_mean",
    "hr_min",
    "walking_hr_avg",
    "spo2_pct",
    "respiratory_rate",
    "log_steps",
    "active_kcal",
    "exercise_min",
]
DEV = ["resting_hr", "hrv_sdnn_ms", "hr_mean"]
BASELINE_DAYS = 60
MIN_HR_SAMPLES = 50
MIN_STEPS = 500


def build_daily_vectors(daily: pd.DataFrame, start: str | None = None) -> pd.DataFrame:
    d = daily.copy()
    d["log_steps"] = np.log1p(d["steps"])
    out = d[ABS].copy()
    for c in DEV:
        # Trailing personal baseline from strictly earlier days.
        base = d[c].shift(1).rolling(BASELINE_DAYS, min_periods=20).median()
        out[f"{c}_dev"] = d[c] - base
    # Only score days the watch was actually worn; partial-wear days are data gaps, not physiology.
    worn = d["resting_hr"].notna() & (d["hr_samples"] >= MIN_HR_SAMPLES) & (d["steps"] >= MIN_STEPS)
    return out[worn].loc[start or first_watch_day(daily) :]


def masked_mse(model, batch):
    x, m = batch
    err = (model(x, m) - x) ** 2 * m
    return err.sum() / m.sum().clamp(min=1)


def run(source, latent: int = 4, quantile: float = 0.98, k_neighbors: int = 5, fig_dir=FIG_DIR, tracker=None) -> dict:
    tracker = tracker or Tracker(enabled=False)
    daily = load_daily(source)
    vec = build_daily_vectors(daily)
    if len(vec) < 120:
        raise NotEnoughData(f"anomaly: only {len(vec)} days with the watch worn (need 120)")
    cols = list(vec.columns)
    # Retrospective, unsupervised detection over the whole history: train on all days except
    # every 7th calendar week, which is held out (as whole weeks) for early stopping. A
    # chronological split would make the later years "anomalous" simply because RHR/HRV
    # levels and the available sensors (e.g. SpO2 only on newer watches) changed.
    week = ((vec.index - vec.index[0]).days // 7).to_numpy()
    va = week % 7 == 3
    tr = ~va
    raw = vec.to_numpy(float)
    mask = np.isfinite(raw).astype(np.float32)
    mu, sd = np.nanmean(raw, 0), np.nanstd(raw, 0)
    # Winsorize at +-4 SD so a single sensor glitch cannot dominate the score.
    z = np.nan_to_num(np.clip((raw - mu) / sd, -4, 4), nan=0.0).astype(np.float32)

    device = get_device()
    set_seed()
    xt, mt = to_tensors(z, mask, device=device)
    model = MaskedAutoencoder(len(cols), latent=latent, hidden=32, dropout=0.1).to(device)
    tr_i, va_i = (torch.as_tensor(np.where(p)[0], device=device) for p in (tr, va))
    res = fit(model, masked_mse, (xt[tr_i], mt[tr_i]), (xt[va_i], mt[va_i]), epochs=400, lr=2e-3, patience=30)
    model.eval()
    with torch.no_grad():
        recon = model(xt, mt).cpu().numpy()
        emb = model.encode(xt, mt).cpu().numpy()

    def normalized_error(rec):
        """Squared error per feature divided by that feature's typical training error, so
        hard-to-reconstruct features (steps) do not dominate easy ones (SpO2)."""
        sq = (rec - z) ** 2 * mask
        typical = sq[tr].sum(0) / np.maximum(mask[tr].sum(0), 1)
        return sq / np.maximum(typical, 1e-6)

    err = normalized_error(recon)
    score = err.sum(1) / np.maximum(mask.sum(1), 1)
    threshold = float(np.quantile(score, quantile))
    is_anom = score > threshold

    # Linear baseline: PCA (fit on the training days) with the same latent size.
    zt = torch.as_tensor(z[tr])
    _, comps, evr = pca(zt, latent)
    mean = zt.mean(0).numpy()
    pca_recon = (z - mean) @ comps.numpy().T @ comps.numpy() + mean
    raw_mse = lambda rec: ((rec - z) ** 2 * mask).sum(1) / np.maximum(mask.sum(1), 1)  # noqa: E731
    val_recon_ae = float(raw_mse(recon)[va].mean())
    val_recon_pca = float(raw_mse(pca_recon)[va].mean())
    pca_score = normalized_error(pca_recon).sum(1) / np.maximum(mask.sum(1), 1)

    resid = z - recon  # positive = the day was higher than the model expected
    top = []
    for i in range(len(z)):
        order = np.argsort(-err[i])[:3]
        top.append(", ".join(f"{cols[j]} {resid[i, j]:+.1f}sd" for j in order if err[i, j] > 0))
    out = pd.DataFrame(
        {
            "local_date": vec.index.date,
            "anomaly_score": score.astype(float),
            "is_anomaly": is_anom,
            "top_features": top,
            "resting_hr": vec["resting_hr"].to_numpy(),
            "hrv_sdnn_ms": vec["hrv_sdnn_ms"].to_numpy(),
            "resting_hr_dev": vec["resting_hr_dev"].to_numpy(),
            "hrv_sdnn_ms_dev": vec["hrv_sdnn_ms_dev"].to_numpy(),
            "steps": daily.loc[vec.index, "steps"].to_numpy(),
            "pca_anomaly_score": pca_score.astype(float),
            "illness_score": _illness_score(vec),
        }
    )
    insight = _illness_like(out, vec)
    episodes = _episodes(out)
    similar, knn_stats = _similar_days(vec.index, emb, daily, k_neighbors)

    figs = _figures(out, threshold, fig_dir)
    metrics = {
        "threshold": threshold,
        "n_days": len(out),
        "n_anomalies": int(is_anom.sum()),
        "best_epoch": res.best_epoch,
        "val_recon_mse_autoencoder": val_recon_ae,
        "val_recon_mse_pca": val_recon_pca,
        "pca_explained_variance": float(evr.sum()),
        "spearman_ae_vs_pca_score": float(pd.Series(score).corr(pd.Series(pca_score), method="spearman")),
        **insight,
        **knn_stats,
    }
    with tracker.run("daily_anomalies"):
        tracker.params({"latent": latent, "quantile": quantile, "features": ",".join(cols)})
        tracker.metrics(metrics)
        tracker.model(model, "daily_autoencoder")
        for f in figs:
            tracker.figure(f)
    return {
        "anomalies": out,
        "similar_days": similar,
        "episodes": episodes,
        "metrics": metrics,
        "figures": figs,
        "embedding": pd.DataFrame(emb, index=vec.index, columns=[f"z{i}" for i in range(latent)]),
    }


def _illness_score(vec: pd.DataFrame) -> np.ndarray:
    """Rule-based complement to the autoencoder: RHR above and HRV below the 60-day baseline,
    each in SD units (z(RHR_dev) - z(HRV_dev)). The AE learns that the two usually move
    together, so it under-scores extreme but consistent illness-like days; this does not."""
    r = vec["resting_hr_dev"] / vec["resting_hr_dev"].std()
    h = vec["hrv_sdnn_ms_dev"] / vec["hrv_sdnn_ms_dev"].std()
    return (r - h).to_numpy(dtype=float)


def _illness_like(out: pd.DataFrame, vec: pd.DataFrame) -> dict:
    """Do anomalies concentrate on 'RHR up + HRV down vs. personal baseline' days?"""
    rhr_sd = vec["resting_hr_dev"].std()
    hrv_sd = vec["hrv_sdnn_ms_dev"].std()
    pattern = (vec["resting_hr_dev"] > rhr_sd).to_numpy() & (vec["hrv_sdnn_ms_dev"] < -0.5 * hrv_sd).to_numpy()
    a = out["is_anomaly"].to_numpy()
    return {
        "illness_like_share_all_days": float(pattern.mean()),
        "illness_like_share_anomalies": float(pattern[a].mean()) if a.any() else float("nan"),
        "illness_like_lift": float(pattern[a].mean() / max(pattern.mean(), 1e-9)) if a.any() else float("nan"),
        "illness_like_auc_autoencoder": _auc(pattern, out["anomaly_score"].to_numpy()),
        "illness_like_auc_pca": _auc(pattern, out["pca_anomaly_score"].to_numpy()),
        "anomaly_mean_rhr_dev": float(out.loc[a, "resting_hr_dev"].mean()),
        "anomaly_mean_hrv_dev": float(out.loc[a, "hrv_sdnn_ms_dev"].mean()),
        "normal_mean_steps": float(out.loc[~a, "steps"].mean()),
        "anomaly_mean_steps": float(out.loc[a, "steps"].mean()),
    }


def _auc(labels: np.ndarray, score: np.ndarray) -> float:
    """How well a score ranks the illness-like days above the rest (0.5 = no better than chance)."""
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(score)
    return float(roc_auc_score(labels[ok], score[ok])) if labels[ok].any() and not labels[ok].all() else float("nan")


def _episodes(out: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """Group anomaly days that are at most ``max_gap`` days apart into episodes."""
    a = out[out["is_anomaly"]].copy()
    if a.empty:
        return pd.DataFrame(columns=["start", "end", "n_days", "max_score", "mean_rhr_dev", "mean_hrv_dev"])
    d = pd.to_datetime(a["local_date"])
    a["episode"] = (d.diff().dt.days.fillna(999) > max_gap).cumsum().to_numpy()
    ep = a.groupby("episode").agg(
        start=("local_date", "min"),
        end=("local_date", "max"),
        n_days=("local_date", "size"),
        max_score=("anomaly_score", "max"),
        mean_rhr_dev=("resting_hr_dev", "mean"),
        mean_hrv_dev=("hrv_sdnn_ms_dev", "mean"),
    )
    return ep.sort_values("max_score", ascending=False).reset_index(drop=True)


def _similar_days(dates: pd.DatetimeIndex, emb: np.ndarray, daily: pd.DataFrame, k: int, exclude_days: int = 3):
    """k nearest days in the autoencoder embedding, excluding the days right around the query."""
    e = torch.as_tensor(emb)
    dist = torch.cdist(e, e)
    day_num = torch.as_tensor(np.asarray((dates - dates[0]).days))
    near = (day_num[:, None] - day_num[None, :]).abs() <= exclude_days
    dist[near] = float("inf")
    d, j = dist.topk(k, largest=False)
    rows = pd.DataFrame(
        {
            "local_date": np.repeat(dates.date, k),
            "neighbor_rank": np.tile(np.arange(1, k + 1), len(dates)),
            "neighbor_date": dates.date[j.numpy().ravel()],
            "distance": d.numpy().ravel().astype(float),
        }
    )
    # Sanity check: neighbours should share context the model never saw directly.
    workout = (daily.loc[dates, "n_workouts"].fillna(0) > 0).to_numpy()
    agree = (workout[j.numpy()] == workout[:, None]).mean()
    rng = np.random.default_rng(SEED)
    rand = rng.integers(0, len(dates), size=j.shape)
    agree_rand = (workout[rand] == workout[:, None]).mean()
    rhr = daily.loc[dates, "resting_hr"].to_numpy()
    rhr_gap = np.nanmean(np.abs(rhr[j.numpy()] - rhr[:, None]))
    rhr_gap_rand = np.nanmean(np.abs(rhr[rand] - rhr[:, None]))
    return rows, {
        "knn_workout_agreement": float(agree),
        "random_workout_agreement": float(agree_rand),
        "knn_rhr_abs_gap": float(rhr_gap),
        "random_rhr_abs_gap": float(rhr_gap_rand),
    }


def _figures(out: pd.DataFrame, threshold: float, fig_dir):
    d = pd.to_datetime(out["local_date"])
    a = out["is_anomaly"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes[0]
    ax.plot(d, out["anomaly_score"], color=GRAY, lw=0.6)
    ax.scatter(d[a], out.loc[a, "anomaly_score"], color=SERIES[7], s=14, zorder=3, label="anomaly (top 2%)")
    ax.axhline(threshold, color=SERIES[7], lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("reconstruction error")
    ax.set_title("Daily autoencoder anomaly score")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(
        d,
        out["resting_hr_dev"].rolling(7, min_periods=1).mean(),
        color=SERIES[1],
        lw=1,
        label="resting HR vs 60-day baseline (bpm, 7d avg)",
    )
    ax.plot(
        d,
        out["hrv_sdnn_ms_dev"].rolling(7, min_periods=1).mean(),
        color=SERIES[0],
        lw=1,
        label="HRV vs 60-day baseline (ms, 7d avg)",
    )
    ax.scatter(d[a], out.loc[a, "resting_hr_dev"], color=SERIES[7], s=10, zorder=3, label="anomaly day RHR dev")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_ylabel("deviation")
    ax.legend(loc="upper left", fontsize=8, ncols=3)
    fig.tight_layout()
    f1 = save(fig, fig_dir / "daily_anomalies.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        out.loc[~a, "resting_hr_dev"], out.loc[~a, "hrv_sdnn_ms_dev"], s=6, color=GRAY, alpha=0.35, label="normal"
    )
    ax.scatter(out.loc[a, "resting_hr_dev"], out.loc[a, "hrv_sdnn_ms_dev"], s=18, color=SERIES[7], label="anomaly")
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("resting HR - 60-day baseline (bpm)")
    ax.set_ylabel("HRV SDNN - 60-day baseline (ms)")
    ax.set_title("Anomalies vs personal baseline\n(lower right = illness/strain-like)")
    ax.legend()
    fig.tight_layout()
    f2 = save(fig, fig_dir / "daily_anomalies_scatter.png")
    return [f1, f2]
