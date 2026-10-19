"""Run every ML experiment locally on the Parquet lakehouse and write outputs, figures and metrics.

    .venv/bin/python ml/run_all.py            # full run (~ a few minutes on M-series / CPU)
    .venv/bin/python ml/run_all.py --quick    # 1 seed, no Chronos download
    .venv/bin/python ml/run_all.py --only forecast anomaly

Outputs: data/ml/*.parquet, ml/reports/figures/*.png, ml/reports/metrics.json, ml/reports/metrics.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from healthml.config import FIG_DIR, OUT_DIR, REPORT_DIR, NotEnoughData, ensure_dirs, get_device
from healthml.experiments import anomaly, classify, clustering, ecg_cnn, forecast, insights, workouts
from healthml.outputs import output_tables
from healthml.sources import ParquetSource
from healthml.tracking import Tracker

EXPERIMENTS = ["forecast", "classify", "anomaly", "ecg", "clustering", "workouts", "insights"]


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return o.reset_index().to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def metrics_markdown(results: dict) -> str:
    lines = ["# Generated metrics (from the last `ml/run_all.py` run)", ""]
    for key in ("forecast_resting_hr", "forecast_hrv_sdnn_ms"):
        if key in results:
            r = results[key]
            lines += [
                f"## {key}",
                "",
                f"{r['info']}",
                "",
                "| model | val MAE | val RMSE | test MAE | test RMSE | test R² | skill vs persistence |",
                "|---|---|---|---|---|---|---|",
            ]
            for name, m in sorted(r["metrics"].items(), key=lambda kv: kv[1].get("test", {}).get("mae", 1e9)):
                v, t = m.get("val", {}), m.get("test", {})
                lines.append(
                    f"| {name} | {_fmt(v.get('mae'))} | {_fmt(v.get('rmse'))} | {_fmt(t.get('mae'))} | {_fmt(t.get('rmse'))} | {_fmt(t.get('r2'))} | {_fmt(t.get('mae_skill_vs_persistence'))} |"
                )
            lines += [
                "",
                "Top linear coefficients (residual units per +1 SD):",
                "",
                r["coefficients"].head(12).to_markdown(index=False),
                "",
                "GBM permutation importance (val):",
                "",
                r["importance"].head(10).to_markdown(index=False),
                "",
            ]
    if "classify" in results:
        for label, r in results["classify"]["results"].items():
            lines += [
                f"## classify {label}",
                "",
                f"train base rate {r['base_rate_train']:.3f}, n={r['n']}",
                "",
                "| model | test AUC | Brier | Brier (base rate) | Brier skill | log loss | log loss (base) | ECE |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for name, m in sorted(r["metrics"].items(), key=lambda kv: -kv[1]["test"]["auc"]):
                t = m["test"]
                lines.append(
                    f"| {name} | {t['auc']:.3f} | {t['brier']:.3f} | {t['brier_base_rate']:.3f} | {t['brier_skill']:.3f} | {t['log_loss']:.3f} | {t['log_loss_base_rate']:.3f} | {t['ece']:.3f} |"
                )
            lines += ["", r["coefficients"].head(10).to_markdown(index=False), ""]
    if "anomaly" in results:
        r = results["anomaly"]
        lines += [
            "## anomaly",
            "",
            json.dumps(r["metrics"], indent=1, default=_jsonable),
            "",
            "Top episodes:",
            "",
            r["episodes"].head(15).to_markdown(index=False),
            "",
            "Top anomaly days:",
            "",
            r["anomalies"]
            .nlargest(15, "anomaly_score")[
                ["local_date", "anomaly_score", "top_features", "resting_hr", "hrv_sdnn_ms", "steps"]
            ]
            .to_markdown(index=False),
            "",
        ]
    if "ecg" in results:
        r = results["ecg"]
        lines += [
            "## ecg",
            "",
            json.dumps(r["metrics"], indent=1),
            "",
            r["recordings"].drop(columns=["polarity"]).to_markdown(index=False, floatfmt=".2f"),
            "",
        ]
    if "clustering" in results:
        r = results["clustering"]
        lines += [
            "## clustering days",
            "",
            json.dumps(r["days"]["metrics"], indent=1),
            "",
            r["days"]["profiles"].to_markdown(index=False, floatfmt=".1f"),
            "",
            "Share per year:",
            "",
            r["days"]["by_year"].rename(columns=r["days"]["names"]).to_markdown(floatfmt=".2f"),
            "",
            "PCA loadings:",
            "",
            r["days"]["pca_components"].to_markdown(floatfmt=".2f"),
            "",
            "## clustering workouts",
            "",
            json.dumps(r["workouts"]["metrics"], indent=1),
            "",
            r["workouts"]["profiles"].to_markdown(index=False, floatfmt=".2f"),
            "",
        ]
    if "workouts" in results:
        for t, r in results["workouts"]["results"].items():
            lines += [
                f"## workouts {t}",
                "",
                f"n={r['n']}, test from {r['test_period_start']}",
                "",
                "| model | val MAE | test MAE | test RMSE | test R² |",
                "|---|---|---|---|---|",
            ]
            for name, m in sorted(r["metrics"].items(), key=lambda kv: kv[1]["test"]["mae"]):
                lines.append(
                    f"| {name} | {m['val']['mae']:.2f} | {m['test']['mae']:.2f} | {m['test']['rmse']:.2f} | {m['test']['r2']:.3f} |"
                )
            lines += ["", r["importance"].head(8).to_markdown(index=False), ""]
    if "insights" in results:
        r = results["insights"]
        lines += [
            "## insights",
            "",
            "Yearly:",
            "",
            r["yearly"].to_markdown(floatfmt=".1f"),
            "",
            f"Slopes per year (since the first watch day): {r['slopes_per_year']}",
            "",
            "Weekday deviation from trailing 28-day mean:",
            "",
            r["weekday"].to_markdown(floatfmt=".2f"),
            "",
            "Month deviation from overall mean (since the first watch day):",
            "",
            r["season"].to_markdown(floatfmt=".2f"),
            "",
            f"Workout effects (diff, 95% CI): {r['workout_effects']}",
            f"n workout days {r['n_workout_days']}, rest days {r['n_rest_days']}",
            "",
            r["load_quartiles"].to_markdown(floatfmt=".2f"),
            "",
            f"corr(RHR, 28d steps) = {r['corr_rhr_vs_28d_steps']:.3f}; corr(28d RHR, 28d load) = {r['corr_28d_rhr_vs_28d_load']:.3f}",
            "",
        ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed and no Chronos")
    ap.add_argument("--only", nargs="*", choices=EXPERIMENTS)
    ap.add_argument("--no-chronos", action="store_true")
    args = ap.parse_args(argv)
    ensure_dirs(OUT_DIR, REPORT_DIR, FIG_DIR)
    todo = args.only or EXPERIMENTS
    n_seeds = 1 if args.quick else 3
    chronos = not (args.quick or args.no_chronos)
    src = ParquetSource()
    tracker = Tracker()
    print(f"device={get_device()} experiments={todo}")
    results, timings = {}, {}
    runners = {
        "forecast_resting_hr": lambda: forecast.run(
            src, target="resting_hr", n_seeds=n_seeds, use_chronos=chronos, tracker=tracker
        ),
        "forecast_hrv_sdnn_ms": lambda: forecast.run(
            src, target="hrv_sdnn_ms", n_seeds=n_seeds, use_chronos=chronos, tracker=tracker
        ),
        "classify": lambda: classify.run(src, tracker=tracker),
        "anomaly": lambda: anomaly.run(src, tracker=tracker),
        "ecg": lambda: ecg_cnn.run(src, tracker=tracker),
        "clustering": lambda: clustering.run(src, tracker=tracker),
        "workouts": lambda: workouts.run(src, tracker=tracker),
        "insights": lambda: insights.run(src),
    }
    skipped = {}
    for name in todo:
        t0 = time.time()
        for key, runner in runners.items():
            if key != name and not key.startswith(f"{name}_"):
                continue
            try:
                results[key] = runner()
            except NotEnoughData as exc:  # too little history in this export: skip, keep the rest
                skipped[key] = str(exc)
                print(f"[{key}] skipped: {exc}")
        timings[name] = round(time.time() - t0, 1)
        print(f"[{name}] done in {timings[name]}s")

    for table, df in output_tables(results).items():
        path = OUT_DIR / f"{table}.parquet"
        df.to_parquet(path, index=False)
        print(f"wrote {path.relative_to(OUT_DIR.parents[1])} ({len(df)} rows)")

    summary = {
        k: {
            kk: vv
            for kk, vv in v.items()
            if kk not in ("predictions", "anomalies", "similar_days", "beats", "embedding")
        }
        for k, v in results.items()
    }
    summary["timings_s"] = timings
    if skipped:
        summary["skipped"] = skipped
    if args.only:  # partial run: merge into the existing metrics files
        old = REPORT_DIR / "metrics.json"
        if old.exists():
            prev = json.loads(old.read_text())
            prev.update(json.loads(json.dumps(summary, default=_jsonable)))
            summary = prev
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    if not args.only:
        (REPORT_DIR / "metrics.md").write_text(metrics_markdown(results))
    else:
        (REPORT_DIR / f"metrics_{'_'.join(todo)}.md").write_text(metrics_markdown(results))
    print(f"total {sum(timings.values()):.0f}s; timings {timings}")


if __name__ == "__main__":
    main()
