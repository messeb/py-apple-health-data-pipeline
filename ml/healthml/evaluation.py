"""Metrics shared by the experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": len(y),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    """AUC plus proper scoring rules, each compared with always predicting the train base rate."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    base = np.full_like(p, base_rate)
    brier, brier_base = brier_score_loss(y, p), brier_score_loss(y, base)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier),
        "brier_base_rate": float(brier_base),
        "brier_skill": float(1 - brier / brier_base),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "log_loss_base_rate": float(log_loss(y, base, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "accuracy@0.5": float(np.mean((p >= 0.5) == y)),
        "positive_rate": float(y.mean()),
        "n": len(y),
    }


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)
