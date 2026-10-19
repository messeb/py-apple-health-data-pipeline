"""(b) Next-day classification: "move ring closed tomorrow" and "workout tomorrow".

The main model is logistic regression in torch; six classic classifiers
(Naive Bayes, decision tree, random forest, gradient boosting, SVM, KNN) run on
the same features and split as a comparison. Every model is scored against always
predicting the training base rate (Brier skill, log loss) and checked for calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..config import DEFAULT_SPLIT, FIG_DIR, SEED, NotEnoughData, TimeSplit, first_watch_day, get_device, set_seed
from ..data import Standardizer, build_classification_frame, load_daily
from ..evaluation import calibration_curve, classification_metrics
from ..models import LogisticRegression
from ..plotting import GRAY, SERIES, plt, save
from ..tracking import Tracker
from ..train import bce_loss, fit, predict, to_tensors

LABELS = {"move_ring_closed": "Move ring closed tomorrow", "had_workout": "Workout tomorrow"}


def _sklearn_models():
    return {
        "naive_bayes_gaussian": GaussianNB(),
        "decision_tree_d4": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED),
        "random_forest": RandomForestClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=SEED
        ),
        "svm_rbf": SVC(C=1.0, probability=True, random_state=SEED),
        "knn_25": KNeighborsClassifier(25),
    }


def run_one(daily: pd.DataFrame, label: str, split: TimeSplit, start: str | None, tracker: Tracker) -> dict:
    frame = build_classification_frame(daily, label)
    start = start or first_watch_day(daily, label if label in daily else "move_ring_closed")
    frame = frame[frame["y"].notna() if start is None else (frame.index >= start) & frame["y"].notna()]
    split = split.resolve(frame["target_date"])
    parts = split.label(frame["target_date"])
    feat_cols = [c for c in frame.columns if c not in ("target_date", "y")]
    x_raw = frame[feat_cols].to_numpy(float)
    y = frame["y"].to_numpy(float)
    tr, va, te = (parts == p for p in ("train", "val", "test"))
    x = Standardizer().fit(x_raw[tr]).transform(x_raw)
    base_rate = float(y[tr].mean())
    if len({*y[tr]}) < 2 or len({*y[te]}) < 2:
        raise NotEnoughData(f"classify {label}: train or test period has a single class")

    device = get_device()
    set_seed()
    xt, yt = to_tensors(x, y, device=device)
    model = LogisticRegression(x.shape[1]).to(device)
    fit(model, bce_loss, (xt[tr], yt[tr]), (xt[va], yt[va]), epochs=500, lr=3e-3, weight_decay=1e-2, patience=30)
    probs = {"logistic_regression_torch": 1 / (1 + np.exp(-predict(model, xt)))}
    for name, clf in _sklearn_models().items():
        clf.fit(x[tr], y[tr])
        probs[name] = clf.predict_proba(x)[:, 1]

    metrics = {
        name: {part: classification_metrics(y[m], p[m], base_rate) for part, m in (("val", va), ("test", te))}
        for name, p in probs.items()
    }
    coef = model.linear.weight.detach().cpu().numpy().ravel()
    coef_df = pd.DataFrame({"feature": feat_cols, "log_odds_per_sd": coef}).sort_values(
        "log_odds_per_sd", key=np.abs, ascending=False
    )
    with tracker.run(f"classify_{label}"):
        tracker.params({"label": label, "base_rate_train": base_rate})
        for name, m in metrics.items():
            tracker.metrics({k: v for k, v in m["test"].items() if k != "n"}, prefix=f"{name}.test.")
    return {
        "metrics": metrics,
        "coefficients": coef_df,
        "base_rate_train": base_rate,
        "y_test": y[te],
        "p_test": probs,
        "test_mask": te,
        "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
    }


def run(source, split: TimeSplit = DEFAULT_SPLIT, start: str | None = None, fig_dir=FIG_DIR, tracker=None) -> dict:
    tracker = tracker or Tracker(enabled=False)
    daily = load_daily(source)
    results = {label: run_one(daily, label, split, start, tracker) for label in LABELS}

    fig, axes = plt.subplots(1, len(results), figsize=(11, 4.4))
    for ax, (label, r) in zip(axes, results.items(), strict=False):
        y = r["y_test"]
        for i, name in enumerate(["logistic_regression_torch", "hist_gradient_boosting", "naive_bayes_gaussian"]):
            p = r["p_test"][name][r["test_mask"]]
            xs, ys, _ = calibration_curve(y, p)
            auc = r["metrics"][name]["test"]["auc"]
            ax.plot(xs, ys, marker="o", ms=5, color=SERIES[i], label=f"{name} (AUC {auc:.2f})")
        ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
        ax.axhline(y.mean(), color=GRAY, lw=1, ls=":", label=f"test positive rate {y.mean():.2f}")
        ax.set_title(f"{LABELS[label]}: calibration (test)")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    figs = [save(fig, fig_dir / "classification_calibration.png")]
    for r in results.values():
        r.pop("p_test"), r.pop("y_test"), r.pop("test_mask")
    return {"results": results, "figures": figs}
