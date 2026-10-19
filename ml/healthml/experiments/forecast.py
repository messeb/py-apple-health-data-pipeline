"""(a) Next-day forecasting of resting HR (or HRV).

Baselines: persistence (tomorrow = today), 7-day mean, seasonal naive (same weekday last
week). Learned models: linear regression (torch), MLP, GRU, LSTM and a small Transformer
over 28-day windows, sklearn histogram gradient boosting, and zero-shot Chronos-Bolt.

All learned models predict the residual ``y(t+1) - mean7(t)`` (z-scored on train), which
is added back to the 7-day mean. The split is chronological and relative to the data
(``TimeSplit``: last 20% of target dates = test, the 15% before = validation for early
stopping, the rest = train). For the test period every learned model is refit on
train+val with the early-stopped epoch count. Neural models are averaged over ``n_seeds`` seeds.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from ..config import (
    DEFAULT_SPLIT,
    FIG_DIR,
    HF_CACHE,
    SEED,
    NotEnoughData,
    TimeSplit,
    first_watch_day,
    get_device,
    set_seed,
)
from ..data import Standardizer, build_sequence_inputs, build_tabular_forecast_frame, load_daily, make_windows
from ..evaluation import regression_metrics
from ..models import MLP, LinearRegressor, RNNForecaster, TransformerForecaster
from ..plotting import GRAY, SERIES, plt, save
from ..tracking import Tracker
from ..train import fit, huber_loss, predict, to_tensors

NON_FEATURES = {"target_date", "y", "y_last", "y_week_ago"}
BASELINES = ["persistence", "mean_7d", "seasonal_naive_7d"]
UNITS = {"resting_hr": "bpm", "hrv_sdnn_ms": "ms"}


def _neural(make_model, x, rz, parts, n_seeds, epochs, lr, wd, refit=True):
    """Train ``n_seeds`` copies with early stopping on val and average their predictions.

    With ``refit`` each seed is retrained on train+val for its early-stopped epoch count and
    that model predicts the test period (the series drifts, so the most recent year matters).
    Train/val predictions always come from the train-only model.
    """
    device = get_device()
    tr, va, te = parts == "train", parts == "val", parts == "test"
    xt, yt = to_tensors(x, rz, device=device)
    tr_i = torch.as_tensor(np.where(tr)[0], device=device)
    va_i = torch.as_tensor(np.where(va)[0], device=device)
    trva_i = torch.as_tensor(np.where(tr | va)[0], device=device)
    outs, epochs_used, last = [], [], None
    for s in range(n_seeds):
        set_seed(SEED + s)
        model = make_model().to(device)
        res = fit(
            model,
            huber_loss,
            (xt[tr_i], yt[tr_i]),
            (xt[va_i], yt[va_i]),
            epochs=epochs,
            lr=lr,
            weight_decay=wd,
            seed=SEED + s,
        )
        pred = predict(model, xt)
        if refit:
            set_seed(SEED + s)
            final = make_model().to(device)
            fit(
                final,
                huber_loss,
                (xt[trva_i], yt[trva_i]),
                (xt[va_i], yt[va_i]),
                epochs=res.best_epoch + 1,
                lr=lr,
                weight_decay=wd,
                seed=SEED + s,
                keep_best=False,
            )
            pred = np.where(te, predict(final, xt), pred)
            model = final
        outs.append(pred)
        epochs_used.append(res.best_epoch)
        last = model
    return np.mean(outs, axis=0), last, epochs_used


def _chronos(series: pd.Series, origin_pos: np.ndarray, context: int = 512) -> np.ndarray | None:
    """Zero-shot one-step median forecast with amazon/chronos-bolt-small (~191 MB). None if unavailable."""
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    try:
        from chronos import BaseChronosPipeline

        pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32
        )
    except Exception as exc:  # library or download missing: skip gracefully
        print(f"[forecast] Chronos skipped: {exc}")
        return None
    values = series.to_numpy(dtype=np.float32)
    out = np.full(len(origin_pos), np.nan)
    for start in range(0, len(origin_pos), 128):
        chunk = origin_pos[start : start + 128]
        ctx = [torch.tensor(values[max(0, p - context + 1) : p + 1]) for p in chunk]
        _, mean = pipe.predict_quantiles(ctx, prediction_length=1, quantile_levels=[0.5])
        out[start : start + len(chunk)] = mean[:, 0].numpy()
    return out


def run(
    source,
    target: str = "resting_hr",
    split: TimeSplit = DEFAULT_SPLIT,
    window: int = 28,
    start: str | None = None,
    n_seeds: int = 3,
    epochs: int = 300,
    use_chronos: bool = True,
    fig_dir=FIG_DIR,
    tracker: Tracker | None = None,
) -> dict:
    tracker = tracker or Tracker(enabled=False)
    t0 = time.time()
    daily = load_daily(source)
    start = start or first_watch_day(daily)
    if start is None or daily[target].notna().sum() < 120:
        raise NotEnoughData(f"forecast {target}: fewer than 120 days with a value")
    tab = build_tabular_forecast_frame(daily, target)
    anchor_all = daily[target].rolling(7, min_periods=3).mean().fillna(tab["y_last"])
    pos = np.arange(len(tab))
    ok = tab["y"].notna() & (tab.index >= start) & (pos >= window - 1) & anchor_all.notna()
    idx = np.where(ok.to_numpy())[0]
    frame = tab.iloc[idx]
    split = split.resolve(frame["target_date"])
    parts = split.label(frame["target_date"])
    tr = parts == "train"
    y = frame["y"].to_numpy(float)
    anchor = anchor_all.iloc[idx].to_numpy(float)

    feat_cols = [c for c in tab.columns if c not in NON_FEATURES]
    x_raw = frame[feat_cols].to_numpy(float)
    x = Standardizer().fit(x_raw[tr]).transform(x_raw)
    resid = y - anchor
    r_mean, r_std = resid[tr].mean(), resid[tr].std()
    rz = ((resid - r_mean) / r_std).astype(np.float32)
    back = lambda z: anchor + r_mean + r_std * np.asarray(z)  # noqa: E731

    seq = build_sequence_inputs(daily)
    seq_train_rows = seq.index < pd.Timestamp(split.val_start)
    seq_x = Standardizer().fit(seq.to_numpy(float)[seq_train_rows]).transform(seq.to_numpy(float))
    windows = make_windows(seq_x, idx, window)

    preds: dict[str, np.ndarray] = {}
    y_last = frame["y_last"].to_numpy(float)
    week_ago = frame["y_week_ago"].to_numpy(float)
    preds["persistence"] = np.where(np.isnan(y_last), anchor, y_last)
    preds["mean_7d"] = anchor
    preds["seasonal_naive_7d"] = np.where(np.isnan(week_ago), anchor, week_ago)

    info: dict = {"n_features_tabular": len(feat_cols), "n_features_sequence": seq.shape[1]}
    n_in, n_seq = x.shape[1], windows.shape[2]
    lin_pred, lin_model, ep = _neural(lambda: LinearRegressor(n_in), x, rz, parts, n_seeds, epochs, 3e-3, 1e-3)
    preds["linear_torch"] = back(lin_pred)
    mlp_pred, _, _ = _neural(lambda: MLP(n_in, (64, 32), 0.3), x, rz, parts, n_seeds, epochs, 1e-3, 1e-2)
    preds["mlp"] = back(mlp_pred)
    for kind in ("gru", "lstm"):
        p, _, _ = _neural(
            lambda kind=kind: RNNForecaster(n_seq, 32, kind=kind, dropout=0.2),
            windows,
            rz,
            parts,
            n_seeds,
            epochs,
            1e-3,
            1e-2,
        )
        preds[f"{kind}_28d"] = back(p)
    p, _, _ = _neural(
        lambda: TransformerForecaster(n_seq, window, d_model=32, nhead=4, layers=2, dropout=0.2),
        windows,
        rz,
        parts,
        n_seeds,
        epochs,
        5e-4,
        1e-2,
    )
    preds["transformer_28d"] = back(p)

    gbm = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=SEED,
    )
    gbm.fit(x_raw[tr], resid[tr])
    gbm_pred = gbm.predict(x_raw)
    gbm_final = clone(gbm).fit(x_raw[tr | (parts == "val")], resid[tr | (parts == "val")])
    preds["hist_gradient_boosting"] = anchor + np.where(parts == "test", gbm_final.predict(x_raw), gbm_pred)

    if use_chronos:
        eval_rows = parts != "train"
        cp = _chronos(daily[target], idx[eval_rows])
        if cp is not None:
            full = np.full(len(y), np.nan)
            full[eval_rows] = cp
            preds["chronos_bolt_small_zero_shot"] = full

    metrics: dict[str, dict] = {}
    for name, p in preds.items():
        metrics[name] = {}
        for part in ("val", "test"):
            m = parts == part
            if np.isfinite(p[m]).all():
                metrics[name][part] = regression_metrics(y[m], p[m])
    persist_test = metrics["persistence"]["test"]["mae"]
    for name in metrics:
        if "test" in metrics[name]:
            metrics[name]["test"]["mae_skill_vs_persistence"] = 1 - metrics[name]["test"]["mae"] / persist_test

    # Interpretable drivers: linear coefficients (residual bpm per +1 SD of feature, averaged
    # weights of the last seed) and GBM permutation importance on the validation year.
    coef = lin_model.linear.weight.detach().cpu().numpy().ravel() * r_std
    coef_df = (
        pd.DataFrame({"feature": feat_cols, "effect_per_sd": coef})
        .assign(abs_effect=lambda d: d.effect_per_sd.abs())
        .sort_values("abs_effect", ascending=False)
        .drop(columns="abs_effect")
    )
    va = parts == "val"
    pi = permutation_importance(
        gbm, x_raw[va], resid[va], n_repeats=5, random_state=SEED, scoring="neg_mean_absolute_error"
    )
    imp_df = pd.DataFrame({"feature": feat_cols, "mae_increase": pi.importances_mean}).sort_values(
        "mae_increase", ascending=False
    )

    test = parts == "test"
    out = pd.concat(
        [
            pd.DataFrame(
                {
                    "local_date": frame["target_date"].to_numpy()[test],
                    "actual": y[test],
                    "predicted": p[test],
                    "model": name,
                }
            )
            for name, p in preds.items()
        ],
        ignore_index=True,
    )
    out["local_date"] = pd.to_datetime(out["local_date"]).dt.date

    figs = _figures(frame["target_date"].to_numpy(), y, preds, parts, metrics, target, fig_dir)
    info.update(
        {
            "n_train": int(tr.sum()),
            "n_val": int(va.sum()),
            "n_test": int(test.sum()),
            "train_period": f"{frame.index[tr].min().date()}..{frame.index[tr].max().date()}",
            "test_period": f"{pd.Timestamp(frame['target_date'][test].min()).date()}.."
            f"{pd.Timestamp(frame['target_date'][test].max()).date()}",
            "linear_best_epochs": ep,
            "runtime_s": round(time.time() - t0, 1),
        }
    )
    with tracker.run(f"forecast_{target}"):
        tracker.params({"target": target, "window": window, "n_seeds": n_seeds, **split.__dict__})
        for name, m in metrics.items():
            if "test" in m:
                tracker.metrics({k: v for k, v in m["test"].items() if k != "n"}, prefix=f"{name}.test.")
        tracker.model(lin_model, f"linear_{target}")
        for f in figs:
            tracker.figure(f)
    return {
        "metrics": metrics,
        "predictions": out,
        "coefficients": coef_df,
        "importance": imp_df,
        "info": info,
        "figures": figs,
    }


def _figures(dates, y, preds, parts, metrics, target, fig_dir):
    unit = UNITS.get(target, "")
    label = {"resting_hr": "resting HR", "hrv_sdnn_ms": "HRV (SDNN)"}.get(target, target)
    names = [n for n in metrics if "test" in metrics[n]]
    learned = [n for n in names if n not in BASELINES]
    best = min(learned, key=lambda n: metrics[n]["test"]["mae"])
    # Last 120 test days: the full period is too dense to read day-level errors.
    test = parts == "test"
    test[: max(0, np.where(test)[0][-1] - 119)] = False
    d = pd.to_datetime(dates[test])

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), height_ratios=[1.3, 1])
    ax = axes[0]
    ax.plot(d, y[test], color=GRAY, lw=1.0, label="actual")
    ax.plot(d, preds["persistence"][test], color=SERIES[1], lw=1.0, alpha=0.8, label="persistence")
    ax.plot(d, preds[best][test], color=SERIES[0], lw=1.8, label=f"{best} (best learned)")
    ax.set_title(f"Next-day {label}: last 120 days of the test period")
    ax.set_ylabel(unit)
    ax.legend(loc="upper left", ncols=3)
    ax = axes[1]
    order = sorted(names, key=lambda n: metrics[n]["test"]["mae"])  # MAE over the whole test period
    vals = [metrics[n]["test"]["mae"] for n in order]
    colors = [GRAY if n in BASELINES else SERIES[0] for n in order]
    ax.barh(order[::-1], vals[::-1], color=colors[::-1], height=0.6)
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel(f"test MAE ({unit}); gray = naive baseline")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return [save(fig, fig_dir / f"forecast_{target}.png")]
