"""Light tests for the ml/healthml package: no time leakage, model shapes, k-means, ECG peaks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from healthml.data import (
    Standardizer,
    build_classification_frame,
    build_sequence_inputs,
    build_tabular_forecast_frame,
    make_windows,
)
from healthml.ecg import detect_r_peaks, hrv_from_peaks
from healthml.models import (
    MLP,
    ConvAutoencoder1D,
    LinearRegressor,
    MaskedAutoencoder,
    RNNForecaster,
    TransformerForecaster,
    kmeans,
    pca,
)


def synthetic_daily(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", name="local_date")
    d = pd.DataFrame(index=idx)
    for c in ["resting_hr", "hrv_sdnn_ms", "hr_mean", "hr_min", "walking_hr_avg"]:
        d[c] = 60 + rng.normal(0, 3, n)
    for c in ["steps", "active_kcal", "exercise_min", "stand_min", "workout_min", "training_load", "flights_climbed"]:
        d[c] = rng.gamma(2, 100, n)
    d["n_workouts"] = rng.integers(0, 2, n).astype(float)
    d["move_ring_closed"] = rng.integers(0, 2, n).astype(float)
    d["move_goal_kcal"] = 600.0
    d.loc[d.index[50:53], "resting_hr"] = np.nan  # a gap to exercise forward filling
    return d


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (build_tabular_forecast_frame, {"target": "resting_hr"}),
        (build_classification_frame, {"label": "move_ring_closed"}),
    ],
)
def test_features_have_no_future_leakage(builder, kwargs):
    d = synthetic_daily()
    cut = d.index[120]
    base = builder(d, **kwargs)
    future = d.copy()
    future.loc[future.index > cut] = future.loc[future.index > cut] * 3 + 17  # change everything after t
    pert = builder(future, **kwargs)
    feats = [c for c in base.columns if c not in ("y", "target_date")]
    upto = base.index <= cut
    pd.testing.assert_frame_equal(base.loc[upto, feats], pert.loc[upto, feats])
    # The label of origin t is the value on t + 1, so it does change for t = cut.
    assert base.loc[cut, "target_date"] == cut + pd.Timedelta(days=1)
    assert base.loc[cut, "y"] != pert.loc[cut, "y"]


def test_label_is_next_day_value():
    d = synthetic_daily()
    f = build_tabular_forecast_frame(d, "resting_hr")
    t = d.index[100]
    assert f.loc[t, "y"] == d.loc[t + pd.Timedelta(days=1), "resting_hr"]
    assert f.loc[t, "y_last"] == d.loc[t, "resting_hr"]
    assert f.loc[t, "y_week_ago"] == d.loc[t - pd.Timedelta(days=6), "resting_hr"]


def test_sequence_windows_only_use_past_rows():
    d = synthetic_daily()
    x = build_sequence_inputs(d).to_numpy(float)
    w = make_windows(x, np.array([27, 100]), 28)
    assert w.shape == (2, 28, x.shape[1])
    np.testing.assert_array_equal(w[1, -1], x[100])
    np.testing.assert_array_equal(w[1, 0], x[73])


def test_standardizer_fits_on_train_only():
    x = np.array([[1.0], [3.0], [100.0]])
    s = Standardizer().fit(x[:2])
    z = s.transform(x)
    assert z[0, 0] == pytest.approx(-1.0) and z[1, 0] == pytest.approx(1.0)
    assert s.transform(np.array([[np.nan]]))[0, 0] == 0.0


def test_forward_shapes():
    b, w, f = 5, 28, 11
    seq = torch.randn(b, w, f)
    flat = torch.randn(b, f)
    assert LinearRegressor(f)(flat).shape == (b,)
    assert MLP(f)(flat).shape == (b,)
    assert RNNForecaster(f, kind="gru")(seq).shape == (b,)
    assert RNNForecaster(f, kind="lstm")(seq).shape == (b,)
    assert TransformerForecaster(f, w)(seq).shape == (b,)
    ae = MaskedAutoencoder(f, latent=3)
    m = torch.ones(b, f)
    assert ae(flat, m).shape == (b, f) and ae.encode(flat, m).shape == (b, 3)
    cae = ConvAutoencoder1D(256)
    beats = torch.randn(b, 1, 256)
    assert cae(beats).shape == (b, 1, 256) and cae.encode(beats).shape == (b, 16)


def test_kmeans_converges_on_toy_blobs():
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [8.0, 8.0], [-8.0, 8.0]])
    x = torch.cat([c + 0.5 * torch.randn(100, 2, generator=g) for c in centers])
    found, labels, inertia = kmeans(x, 3, n_init=5, seed=0)
    # Every true blob maps to exactly one cluster and centers are recovered.
    for i in range(3):
        assert len(set(labels[i * 100 : (i + 1) * 100].tolist())) == 1
    dist = torch.cdist(centers, found).min(dim=1).values
    assert (dist < 0.3).all()
    assert inertia < 300 * 0.5**2 * 2 * 1.5


def test_pca_recovers_main_axis():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(500, 1, generator=g)
    x = torch.cat([t * 5, t * 5 + 0.1 * torch.randn(500, 1, generator=g)], dim=1)
    _, comps, evr = pca(x, 1)
    assert evr[0] > 0.99
    assert abs(abs(comps[0, 0].item()) - 2**-0.5) < 0.01


def test_r_peak_detection_on_synthetic_ecg():
    fs, hr = 512.0, 75
    t = np.arange(int(30 * fs)) / fs
    rr = 60 / hr
    x = np.zeros_like(t)
    for k in np.arange(0.5, 30, rr):
        x += 1000 * np.exp(-((t - k) ** 2) / (2 * 0.01**2))  # R
        x += 200 * np.exp(-((t - k - 0.25) ** 2) / (2 * 0.04**2))  # T
    peaks, pol = detect_r_peaks(x, fs)
    assert pol == 1
    stats = hrv_from_peaks(peaks, fs)
    assert stats["hr_bpm"] == pytest.approx(hr, abs=1)
    assert len(peaks) == pytest.approx(29 * hr / 60, abs=2)
    peaks_neg, pol_neg = detect_r_peaks(-x, fs)
    assert pol_neg == -1 and len(peaks_neg) == len(peaks)


def test_time_split_is_relative_to_the_data():
    from healthml.config import NotEnoughData, TimeSplit

    dates = pd.date_range("2030-01-01", periods=1000, freq="D")
    split = TimeSplit(val_start=None, test_start=None).resolve(dates)
    parts = split.label(dates)
    assert (parts == "test").sum() == 200 and (parts == "val").sum() == 150 and (parts == "train").sum() == 650
    assert pd.Timestamp(split.test_start) == dates[800] and pd.Timestamp(split.val_start) == dates[650]
    # Explicit dates (e.g. HEALTHML_VAL_START / HEALTHML_TEST_START) win.
    pinned = TimeSplit(val_start="2031-01-01", test_start="2032-01-01")
    assert pinned.label(dates)[365] == "val" and pinned.label(dates)[731] == "test"
    with pytest.raises(NotEnoughData):
        TimeSplit(val_start=None, test_start=None).resolve(dates[:60])


def test_run_all_quick_on_synthetic_lakehouse(synthetic_data_dir):
    """`ml/run_all.py --quick` ran on the synthetic export (conftest) with every experiment and output."""
    import json

    ml = synthetic_data_dir / "ml"
    expected = {
        "forecast_resting_hr",
        "forecast_hrv_sdnn_ms",
        "forecast_metrics",
        "daily_anomalies",
        "similar_days",
        "anomaly_episodes",
        "day_clusters",
        "day_cluster_profiles",
        "workout_clusters",
        "ecg_beat_anomalies",
        "ecg_recordings",
    }
    assert expected <= {p.stem for p in ml.glob("*.parquet")}
    metrics = json.loads((synthetic_data_dir.parent / "reports" / "metrics.json").read_text())
    assert "skipped" not in metrics, metrics.get("skipped")
    test = pd.read_parquet(ml / "forecast_metrics.parquet").query("split == 'test' and target == 'resting_hr'")
    assert test.set_index("model").loc["persistence", "mae"] < 5
    fc = pd.read_parquet(ml / "forecast_resting_hr.parquet")
    assert pd.to_datetime(fc["local_date"]).min() > pd.Timestamp("2026-01-01")  # last ~20% of a 400-day export
