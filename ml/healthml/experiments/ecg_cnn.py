"""(d) 1D-CNN autoencoder on single ECG heartbeats.

R peaks are found with scipy ``find_peaks`` on the band-passed signal; each beat
(-250 ms .. +450 ms around R) is resampled to 256 samples and scaled by the recording's
median R amplitude. The convolutional autoencoder is trained on beats from the earlier
"Sinus Rhythm" recordings (chronological split by recording, never by beat, so beats of
one recording are not in train and val at the same time). The "Inconclusive" recording is
never trained on and serves as a sanity check. PCA reconstruction is the linear baseline.

A beat is flagged (``is_anomaly``) for its shape (score above the 99th percentile of
train/val beats; ``anomaly_score`` is the reconstruction error divided by the median
training beat's error) and/or its rhythm (preceding RR more than 25% off the recording's
median RR). ``flag_reason`` says which: 'shape', 'rr_interval' or 'shape+rr_interval'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import FIG_DIR, NotEnoughData, get_device, set_seed
from ..ecg import BEAT_AFTER_S, BEAT_BEFORE_S, BEAT_LEN, bandpass, detect_r_peaks, hrv_from_peaks, segment_beats
from ..models import ConvAutoencoder1D, pca
from ..plotting import GRAY, SERIES, plt, save
from ..tracking import Tracker
from ..train import fit, to_tensors

RR_TOLERANCE = 0.25


def ae_loss(model, batch):
    (x,) = batch
    return nn.functional.mse_loss(model(x), x)


def prepare(source):
    meta = source.load("gold_ecg_summary")
    meta["recorded_local"] = pd.to_datetime(meta["recorded_local"])
    meta = meta.sort_values("recorded_local").reset_index(drop=True)
    if meta.empty:
        raise NotEnoughData("ecg: no ECG recordings")
    samples = source.load("silver_ecg_samples")
    beats, rows, rec_rows, signals = [], [], [], {}
    for _, m in meta.iterrows():
        s = samples[samples["ecg_id"] == m.ecg_id].sort_values("sample_index")["value_uv"].to_numpy(float)
        fs = float(m.sample_rate_hz)
        x = bandpass(s, fs)
        peaks, pol = detect_r_peaks(x, fs)
        b, kept = segment_beats(pol * x, peaks, fs)
        amp = np.median(b[:, int(BEAT_LEN * BEAT_BEFORE_S / (BEAT_BEFORE_S + BEAT_AFTER_S))]) if len(b) else 1.0
        b = b / max(amp, 1e-6)
        rr_prev = np.r_[np.nan, np.diff(peaks) / fs * 1000.0]
        rr_map = dict(zip(peaks, rr_prev, strict=False))
        for i, p in enumerate(kept):
            rows.append(
                {
                    "ecg_id": m.ecg_id,
                    "recorded_local": m.recorded_local,
                    "beat_index": i,
                    "r_peak_s": p / fs,
                    "rr_ms": rr_map[p],
                }
            )
        beats.append(b)
        rec_rows.append(
            {
                "ecg_id": m.ecg_id,
                "recorded_local": m.recorded_local,
                "classification": m.classification,
                "polarity": pol,
                "n_beats": len(kept),
                **hrv_from_peaks(peaks, fs),
            }
        )
        signals[m.ecg_id] = (x * pol, peaks, fs)
    return np.concatenate(beats).astype(np.float32), pd.DataFrame(rows), pd.DataFrame(rec_rows), signals


def run(source, latent: int = 16, quantile: float = 0.99, fig_dir=FIG_DIR, tracker=None) -> dict:
    tracker = tracker or Tracker(enabled=False)
    beats, beat_df, rec, signals = prepare(source)
    sinus = rec[rec["classification"].str.startswith("Sinus")]["ecg_id"].tolist()
    if len(sinus) < 2:
        raise NotEnoughData(f"ecg: {len(sinus)} sinus-rhythm recordings (need 2 for train/val)")
    n_tr = round(0.7 * len(sinus))
    n_va = max(1, round(0.15 * len(sinus)))
    role = {e: "train" for e in sinus[:n_tr]} | {e: "val" for e in sinus[n_tr : n_tr + n_va]}
    rec["split"] = rec["ecg_id"].map(role).fillna("test")
    beat_df["split"] = beat_df["ecg_id"].map(dict(zip(rec.ecg_id, rec.split, strict=False)))
    tr = (beat_df["split"] == "train").to_numpy()
    va = (beat_df["split"] == "val").to_numpy()

    device = get_device()
    set_seed()
    (xt,) = to_tensors(beats[:, None, :], device=device)
    model = ConvAutoencoder1D(BEAT_LEN, (16, 32, 32), latent).to(device)
    tr_i, va_i = (torch.as_tensor(np.where(m)[0], device=device) for m in (tr, va))
    res = fit(model, ae_loss, (xt[tr_i],), (xt[va_i],), epochs=300, lr=2e-3, batch_size=64, patience=30)
    with torch.no_grad():
        recon = torch.cat([model(xt[i : i + 512]) for i in range(0, len(xt), 512)]).cpu().numpy()[:, 0]
    score = ((recon - beats) ** 2).mean(1)

    bt = torch.as_tensor(beats[tr])
    _, comps, _ = pca(bt, latent)
    mean = bt.mean(0).numpy()
    pca_recon = (beats - mean) @ comps.numpy().T @ comps.numpy() + mean
    pca_score = ((pca_recon - beats) ** 2).mean(1)

    # Score in units of a typical training beat's error (1.0 = median train beat), so the
    # values are readable; the raw MSE is kept as reconstruction_mse.
    typical = float(np.median(score[tr]))
    rel = score / typical
    threshold = float(np.quantile(rel[tr | va], quantile))
    beat_df["anomaly_score"] = rel.astype(float)
    beat_df["reconstruction_mse"] = score.astype(float)
    shape_flag = rel > threshold
    # Rhythm flag: RR interval before the beat more than 25% away from the recording's median RR
    # (premature / missed beat or detection artifact). Independent of the beat shape.
    rr_med = beat_df.groupby("ecg_id")["rr_ms"].transform("median")
    rr_flag = ((beat_df["rr_ms"] - rr_med).abs() > RR_TOLERANCE * rr_med).to_numpy()
    beat_df["is_anomaly"] = shape_flag | rr_flag
    beat_df["flag_reason"] = np.select(
        [shape_flag & rr_flag, shape_flag, rr_flag], ["shape+rr_interval", "shape", "rr_interval"], default=None
    )
    per_rec = beat_df.groupby("ecg_id").agg(
        mean_beat_score=("anomaly_score", "mean"),
        max_beat_score=("anomaly_score", "max"),
        n_anomalous_beats=("is_anomaly", "sum"),
        n_shape_flags=("flag_reason", lambda r: int(r.isin(["shape", "shape+rr_interval"]).sum())),
        n_rr_flags=("flag_reason", lambda r: int(r.isin(["rr_interval", "shape+rr_interval"]).sum())),
    )
    rec = rec.merge(per_rec, on="ecg_id", how="left")
    rec["score_rank"] = rec["mean_beat_score"].rank(ascending=False).astype(int)

    metrics = {
        "n_recordings": len(rec),
        "n_beats": len(beat_df),
        "best_epoch": res.best_epoch,
        "val_mse_cnn_ae": float(score[va].mean()),
        "val_mse_pca": float(pca_score[va].mean()),
        "test_mse_cnn_ae": float(score[~(tr | va)].mean()),
        "test_mse_pca": float(pca_score[~(tr | va)].mean()),
        "threshold_x_typical_beat": threshold,
        "typical_beat_mse": typical,
        "n_anomalous_beats": int(beat_df["is_anomaly"].sum()),
        "n_shape_flags": int(shape_flag.sum()),
        "n_rr_flags": int(rr_flag.sum()),
        "ecg_hr_mean_bpm": float(rec["hr_bpm"].mean()),
        "ecg_rmssd_median_ms": float(rec["rmssd_ms"].median()),
    }
    inc = rec[~rec["classification"].str.startswith("Sinus")]
    if len(inc):
        metrics["inconclusive_score_rank"] = int(inc["score_rank"].iloc[0])
    figs = _figures(beats, recon, beat_df, rec, signals, fig_dir)
    with tracker.run("ecg_beat_autoencoder"):
        tracker.params({"latent": latent, "beat_len": BEAT_LEN, "quantile": quantile})
        tracker.metrics(metrics)
        tracker.model(model, "ecg_conv_autoencoder")
        for f in figs:
            tracker.figure(f)
    beat_df["recorded_local"] = pd.to_datetime(beat_df["recorded_local"])
    return {"beats": beat_df, "recordings": rec, "metrics": metrics, "figures": figs}


def _figures(beats, recon, beat_df, rec, signals, fig_dir):
    t_ms = np.linspace(-BEAT_BEFORE_S, BEAT_AFTER_S, BEAT_LEN) * 1000
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), width_ratios=[1.6, 1, 1])
    eid = rec.sort_values("recorded_local")["ecg_id"].iloc[-1]
    x, peaks, fs = signals[eid]
    n = int(10 * fs)
    t = np.arange(n) / fs
    ax = axes[0]
    ax.plot(t, x[:n], color=SERIES[0], lw=0.8)
    pk = peaks[peaks < n]
    ax.scatter(pk / fs, x[pk], color=SERIES[7], s=16, zorder=3, label="detected R peak")
    ax.set_title(f"{eid}: first 10 s, band-passed")
    ax.set_xlabel("s")
    ax.set_ylabel("µV")
    ax.legend(loc="upper right")
    ax = axes[1]
    normal = beats[~beat_df["flag_reason"].isin(["shape", "shape+rr_interval"]).to_numpy()]
    med, lo, hi = np.median(normal, 0), np.percentile(normal, 5, 0), np.percentile(normal, 95, 0)
    ax.fill_between(t_ms, lo, hi, color=SERIES[0], alpha=0.18, lw=0, label="5-95% of normal beats")
    ax.plot(t_ms, med, color=SERIES[0], lw=1.6, label="median beat")
    ax.set_title("Beat template (all recordings)")
    ax.set_xlabel("ms from R peak")
    ax.set_ylabel("x median R amplitude")
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[2]
    i = int(np.argmax(beat_df["anomaly_score"].to_numpy()))
    ax.plot(t_ms, beats[i], color=SERIES[7], lw=1.4, label="most anomalous beat")
    ax.plot(t_ms, recon[i], color=GRAY, lw=1.2, ls="--", label="CNN reconstruction")
    ax.plot(t_ms, med, color=SERIES[0], lw=0.9, alpha=0.6, label="median beat")
    r = beat_df.iloc[i]
    ax.set_title(f"{r.ecg_id}, beat {r.beat_index}")
    ax.set_xlabel("ms from R peak")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return [save(fig, fig_dir / "ecg_beats.png")]
