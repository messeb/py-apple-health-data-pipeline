"""Descriptive statistics behind the plain-language insights: trends, weekday and seasonal
effects, and the association of workouts with next-day HRV / resting HR.

Effects are measured as deviations from a trailing 28-day personal baseline, so multi-year
fitness drift does not masquerade as a weekday or workout effect. These are associations
for one person, not causal estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FIG_DIR, SEED, NotEnoughData, first_watch_day
from ..data import load_daily
from ..plotting import GRAY, SERIES, plt, save

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _dev(s: pd.Series, days: int = 28) -> pd.Series:
    return s - s.shift(1).rolling(days, min_periods=10).mean()


def _boot_diff(a: np.ndarray, b: np.ndarray, n: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run(source, fig_dir=FIG_DIR, tracker=None) -> dict:
    d = load_daily(source)
    yearly = d.groupby("year").agg(
        resting_hr=("resting_hr", "median"),
        hrv_sdnn_ms=("hrv_sdnn_ms", "median"),
        steps=("steps", "median"),
        active_kcal=("active_kcal", "median"),
        exercise_min=("exercise_min", "median"),
        vo2max=("vo2max", "median"),
        workouts=("n_workouts", "sum"),
        cycling_km=("cycling_km", "sum"),
        walking_speed_kmh=("walking_speed_kmh", "median"),
    )
    yearly["days"] = d.groupby("year")["steps"].count()

    # Physiology (RHR, HRV) only from the first watch day; steps exist from the phone era.
    watch_start = first_watch_day(d)
    if watch_start is None:
        raise NotEnoughData("insights: no resting heart rate in the data")
    phys = d.loc[watch_start:]
    slopes = {}
    for c in ("resting_hr", "hrv_sdnn_ms"):
        s = phys[c].dropna()
        t = (s.index - s.index[0]).days / 365.25
        slopes[c] = float(np.polyfit(t, s.to_numpy(), 1)[0])

    dev = pd.DataFrame({c: _dev(phys[c]) for c in ("resting_hr", "hrv_sdnn_ms", "steps", "active_kcal")})
    dev["dow"] = phys["day_of_week"]
    dev["month"] = phys["month"]
    weekday = dev.groupby("dow")[["resting_hr", "hrv_sdnn_ms", "steps", "active_kcal"]].mean()
    weekday.index = DOW
    raw_month = phys.groupby("month")[["resting_hr", "hrv_sdnn_ms", "steps"]].mean()
    season = raw_month - phys[["resting_hr", "hrv_sdnn_ms", "steps"]].mean()

    # Workout today -> deviation of tomorrow's HRV / RHR from the baseline before today.
    base_hrv = phys["hrv_sdnn_ms"].rolling(28, min_periods=10).mean()
    base_rhr = phys["resting_hr"].rolling(28, min_periods=10).mean()
    nxt = pd.DataFrame(
        {
            "hrv_next_dev": phys["hrv_sdnn_ms"].shift(-1) - base_hrv,
            "rhr_next_dev": phys["resting_hr"].shift(-1) - base_rhr,
            "workout": phys["n_workouts"] > 0,
            "load": phys["training_load"],
            "dow_next": (phys["day_of_week"] + 1) % 7,
        }
    ).dropna(subset=["hrv_next_dev", "rhr_next_dev"])
    effects = {}
    for col in ("hrv_next_dev", "rhr_next_dev"):
        effects[col] = _boot_diff(nxt.loc[nxt.workout, col].to_numpy(), nxt.loc[~nxt.workout, col].to_numpy())
        # Same contrast adjusted for the weekday of the next day (OLS with weekday dummies).
        xd = pd.get_dummies(nxt["dow_next"], prefix="d", drop_first=True).astype(float)
        X = np.column_stack([np.ones(len(nxt)), nxt["workout"].astype(float), xd.to_numpy()])
        beta = np.linalg.lstsq(X, nxt[col].to_numpy(), rcond=None)[0]
        effects[f"{col}_dow_adjusted"] = float(beta[1])
    wk = nxt[nxt.workout & (nxt.load > 0)]
    load_q = wk.groupby(pd.qcut(wk["load"], 4, labels=["Q1 light", "Q2", "Q3", "Q4 hard"]), observed=True)[
        ["hrv_next_dev", "rhr_next_dev"]
    ].mean()
    load_q.loc["no workout"] = nxt.loc[~nxt.workout, ["hrv_next_dev", "rhr_next_dev"]].mean()
    corr_steps_rhr = float(phys["resting_hr"].corr(phys["steps"].rolling(28).mean()))
    corr_load_rhr = float(phys["resting_hr"].rolling(28).mean().corr(phys["training_load"].rolling(28).mean()))

    figs = _figures(yearly, weekday, season, load_q, fig_dir, pd.Timestamp(watch_start).year)
    return {
        "yearly": yearly,
        "slopes_per_year": slopes,
        "weekday": weekday,
        "season": season,
        "workout_effects": effects,
        "load_quartiles": load_q,
        "n_workout_days": int(nxt.workout.sum()),
        "n_rest_days": int((~nxt.workout).sum()),
        "corr_rhr_vs_28d_steps": corr_steps_rhr,
        "corr_28d_rhr_vs_28d_load": corr_load_rhr,
        "figures": figs,
    }


def _figures(yearly, weekday, season, load_q, fig_dir, watch_year):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    y = yearly.loc[watch_year:]
    for ax, col, color, title in (
        (axes[0, 0], "resting_hr", SERIES[7], "Median resting HR per year (bpm)"),
        (axes[0, 1], "hrv_sdnn_ms", SERIES[0], "Median HRV SDNN per year (ms)"),
    ):
        ax.plot(y.index, y[col], marker="o", ms=6, color=color)
        for xx, v in zip(y.index, y[col], strict=False):
            ax.annotate(
                f"{v:.0f}", (xx, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color="#52514e"
            )
        ax.set_title(title)
    ax = axes[0, 2]
    yy = yearly
    ax.bar(yy.index.astype(str), yy["cycling_km"], color=SERIES[2], width=0.6)
    ax.set_title("Cycling km per year")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="x", visible=False)
    ax = axes[1, 0]
    ax.bar(
        weekday.index,
        weekday["resting_hr"],
        color=[SERIES[7] if v > 0 else SERIES[0] for v in weekday["resting_hr"]],
        width=0.6,
    )
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_title("Resting HR by weekday\n(vs trailing 28-day mean, bpm)")
    ax.grid(axis="x", visible=False)
    ax = axes[1, 1]
    ax.bar(weekday.index, weekday["steps"], color=SERIES[2], width=0.6)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_title("Steps by weekday\n(vs trailing 28-day mean)")
    ax.grid(axis="x", visible=False)
    ax = axes[1, 2]
    lq = load_q
    xs = np.arange(len(lq))
    ax.bar(xs - 0.18, lq["hrv_next_dev"], width=0.36, color=SERIES[0], label="next-day HRV dev (ms)")
    ax.bar(xs + 0.18, lq["rhr_next_dev"], width=0.36, color=SERIES[7], label="next-day RHR dev (bpm)")
    ax.set_xticks(xs, lq.index, fontsize=8)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_title("Next day vs 28-day baseline,\nby today's training load")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return [save(fig, fig_dir / "insights_overview.png")]
