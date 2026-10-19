"""Feature building on the gold tables.

Convention for every forecasting/classification frame: a row is indexed by the
*origin* day ``t``. All features use data from days ``<= t`` only; the label is the
value on ``t + 1`` and ``target_date`` holds that day. ``tests/test_ml.py`` checks this
by perturbing future values and asserting the features do not change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Plausibility limits; values outside are sensor glitches, not physiology.
CLIP = {
    "resting_hr": (35, 120),
    "hrv_sdnn_ms": (5, 200),
    "hr_mean": (35, 180),
    "hr_min": (30, 150),
    "walking_hr_avg": (40, 180),
    "spo2_pct": (80, 100),
    "respiratory_rate": (10, 30),
}

BOOL_COLS = ["move_ring_closed", "exercise_ring_closed", "stand_ring_closed", "is_weekend"]

# Per-day inputs for the forecasting models.
PHYSIO = ["resting_hr", "hrv_sdnn_ms", "hr_mean", "hr_min", "walking_hr_avg"]
ACTIVITY = ["steps", "active_kcal", "exercise_min", "stand_min", "workout_min", "training_load", "flights_climbed"]


def load_daily(source, start: str | None = None) -> pd.DataFrame:
    """gold_daily_summary on a continuous daily calendar (missing days become NaN rows)."""
    df = source.load("gold_daily_summary")
    df["local_date"] = pd.to_datetime(df["local_date"])
    df = df.sort_values("local_date").drop_duplicates("local_date")
    for c in BOOL_COLS:
        if c in df:
            df[c] = df[c].map({True: 1.0, False: 0.0, "true": 1.0, "false": 0.0}).astype(float)
    for c, (lo, hi) in CLIP.items():
        if c in df:
            df[c] = df[c].where(df[c].between(lo, hi))
    df = df.set_index("local_date").asfreq("D")
    df.index.name = "local_date"
    df["day_of_week"] = df.index.dayofweek  # 0 = Monday
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    if start:
        df = df.loc[start:]
    return df


def calendar_features(dates: pd.DatetimeIndex, prefix: str = "") -> pd.DataFrame:
    """Day-of-week one-hot and a smooth annual cycle (known in advance, so no leakage)."""
    dow = pd.get_dummies(dates.dayofweek, prefix=f"{prefix}dow").astype(float)
    dow = dow.reindex(columns=[f"{prefix}dow_{i}" for i in range(7)], fill_value=0.0)
    doy = dates.dayofyear.to_numpy()
    out = pd.DataFrame(
        {
            f"{prefix}doy_sin": np.sin(2 * np.pi * doy / 365.25),
            f"{prefix}doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=dates,
    )
    dow.index = dates
    return pd.concat([dow, out], axis=1)


def fill_daily_inputs(daily: pd.DataFrame, cols: list[str], ffill_limit: int = 3) -> pd.DataFrame:
    """Forward fill short gaps (past values only) and add a missing indicator per column."""
    x = daily[cols].copy()
    miss = x.isna().astype(float).add_suffix("_missing")
    x = x.ffill(limit=ffill_limit)
    return pd.concat([x, miss], axis=1)


def build_tabular_forecast_frame(
    daily: pd.DataFrame, target: str = "resting_hr", relative: bool = True
) -> pd.DataFrame:
    """One row per origin day t: lags/rolling stats up to t, calendar of t+1, label y = target(t+1).

    With ``relative`` the physiological level features are expressed relative to their own
    trailing 90-day mean, so the model learns short-term dynamics instead of memorising the
    absolute level of the training years (resting HR and HRV drift by several units per era).
    """
    base = fill_daily_inputs(daily, PHYSIO + ACTIVITY)
    y_hist = daily[target]
    level = {c: daily[c].rolling(90, min_periods=20).mean() if relative else 0.0 for c in PHYSIO}
    f = pd.DataFrame(index=daily.index)
    for lag in range(7):
        f[f"{target}_lag{lag}"] = base[target].shift(lag) - level[target]
    for w in (7, 28):
        f[f"{target}_mean{w}"] = y_hist.rolling(w, min_periods=max(3, w // 3)).mean() - level[target]
        f[f"{target}_std{w}"] = y_hist.rolling(w, min_periods=max(3, w // 3)).std()
    other = [c for c in PHYSIO if c != target]
    for c in other:
        f[f"{c}_lag0"] = base[c] - level[c]
        f[f"{c}_mean7"] = daily[c].rolling(7, min_periods=3).mean() - level[c]
    for c in ACTIVITY:
        f[f"{c}_lag0"] = base[c]
        f[f"{c}_mean7"] = daily[c].rolling(7, min_periods=3).mean()
    f[f"{target}_missing"] = base[f"{target}_missing"]
    target_dates = daily.index + pd.Timedelta(days=1)
    cal = calendar_features(target_dates, prefix="next_")
    cal.index = daily.index
    f = pd.concat([f, cal], axis=1)
    f["target_date"] = target_dates
    f["y"] = y_hist.shift(-1)
    f["y_last"] = y_hist  # raw value on t, used by the persistence baseline
    f["y_week_ago"] = y_hist.shift(6)  # value on (t+1) - 7, the seasonal-naive baseline
    return f


def build_sequence_inputs(daily: pd.DataFrame, relative: bool = True) -> pd.DataFrame:
    """Per-day feature vectors for the sequence models (filled inputs + own-day calendar).

    ``relative`` subtracts the trailing 90-day mean from the physiological columns, as in
    :func:`build_tabular_forecast_frame`.
    """
    base = fill_daily_inputs(daily, PHYSIO + ACTIVITY)
    if relative:
        for c in PHYSIO:
            base[c] = base[c] - daily[c].rolling(90, min_periods=20).mean()
    cal = calendar_features(daily.index)
    return pd.concat([base, cal], axis=1)


def make_windows(x: np.ndarray, origin_idx: np.ndarray, window: int) -> np.ndarray:
    """Stack x[i-window+1 : i+1] for each origin index i (all past-or-present rows)."""
    return np.stack([x[i - window + 1 : i + 1] for i in origin_idx])


class Standardizer:
    """NaN-aware z-scoring fit on the training rows only; NaN becomes 0 (the train mean)."""

    def fit(self, x: np.ndarray) -> Standardizer:
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)
        self.std_ = np.where(~np.isfinite(self.std_) | (self.std_ < 1e-8), 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean_) / self.std_
        return np.nan_to_num(z, nan=0.0).astype(np.float32)

    def inverse(self, z: np.ndarray, col: int | None = None) -> np.ndarray:
        if col is None:
            return z * self.std_ + self.mean_
        return z * self.std_[col] + self.mean_[col]


def build_classification_frame(daily: pd.DataFrame, label: str = "move_ring_closed") -> pd.DataFrame:
    """Features up to t, label = ``label`` on t+1 (e.g. move ring closed tomorrow / workout tomorrow)."""
    d = daily.copy()
    d["had_workout"] = (d["n_workouts"] > 0).astype(float).where(d["n_workouts"].notna())
    src = d[label]
    f = pd.DataFrame(index=d.index)
    for lag in range(7):
        f[f"{label}_lag{lag}"] = src.shift(lag)
    for w in (7, 28):
        f[f"{label}_rate{w}"] = src.rolling(w, min_periods=w // 2).mean()
    other = "had_workout" if label != "had_workout" else "move_ring_closed"
    f[f"{other}_rate7"] = d[other].rolling(7, min_periods=3).mean()
    f[f"{other}_lag0"] = d[other]
    for c in ["active_kcal", "exercise_min", "steps", "workout_min", "resting_hr", "hrv_sdnn_ms"]:
        f[f"{c}_lag0"] = d[c]
        f[f"{c}_mean7"] = d[c].rolling(7, min_periods=3).mean()
    f["move_goal_kcal"] = d["move_goal_kcal"].ffill()
    f["active_kcal_over_goal7"] = (d["active_kcal"] / d["move_goal_kcal"]).rolling(7, min_periods=3).mean()
    target_dates = d.index + pd.Timedelta(days=1)
    cal = calendar_features(target_dates, prefix="next_")
    cal.index = d.index
    f = pd.concat([f, cal], axis=1)
    f["target_date"] = target_dates
    f["y"] = src.shift(-1)
    return f
