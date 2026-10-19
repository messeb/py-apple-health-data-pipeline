"""Streamlit app tests: data layer (local backend), SQL portability to Databricks,
every page through Streamlit's AppTest, and a real server health check.

Everything runs on the synthetic lakehouse from `conftest.py`, never on `data/`."""

from __future__ import annotations

import datetime as dt
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pytest
import sqlglot

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

import data

on_synthetic_data = pytest.mark.usefixtures("lakehouse")

PAGES = [
    "views/overview.py",
    "views/activity.py",
    "views/heart.py",
    "views/workouts.py",
    "views/body.py",
    "views/sleep.py",
    "views/ecg.py",
    "views/ml.py",
]


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    monkeypatch.setenv("HEALTH_BACKEND", "local")
    monkeypatch.delenv("HEALTH_DATA_DIR", raising=False)


@pytest.fixture
def lakehouse(synthetic_data_dir, monkeypatch) -> Path:
    monkeypatch.setenv("HEALTH_DATA_DIR", str(synthetic_data_dir))
    return synthetic_data_dir


# --------------------------------------------------------------------------- data layer


@on_synthetic_data
def test_local_query_functions_return_data():
    lo, hi = data.date_bounds()
    assert lo < hi
    year_ago = hi - dt.timedelta(days=364)

    daily = data.daily_summary(year_ago, hi)
    assert len(daily) > 300 and {"steps", "resting_hr", "hrv_sdnn_ms"} <= set(daily.columns)
    assert pd.api.types.is_datetime64_any_dtype(daily["local_date"])
    assert not data.coverage().empty
    assert not data.weekly_summary(year_ago, hi).empty

    for metric in data.HOURLY_METRICS:
        prof = data.hourly_activity_profile(metric, year_ago, hi)
        assert not prof.empty, metric
        assert prof["day_of_week"].between(1, 7).all() and prof["local_hour"].between(0, 23).all()
    hr = data.hourly_heart_rate_profile(year_ago, hi)
    assert not hr.empty and hr["hr_mean"].between(30, 200).all()

    workouts = data.workouts(lo, hi)
    assert len(workouts) > 100
    hr_ref = data.workout_hr_reference()
    assert hr_ref is not None and 120 < hr_ref < 230
    route = workouts["route_file"].dropna().iloc[0]
    pts = data.route_points(route)
    assert not pts.empty and pts["lat"].between(-90, 90).all()

    assert not data.sleep(lo, hi).empty
    ecgs = data.ecg_list()
    assert not ecgs.empty
    samples = data.ecg_samples(ecgs["ecg_id"].iloc[0])
    assert len(samples) > 1000

    assert isinstance(data.list_ml_tables(), list)
    for stem in data.list_ml_tables():
        assert isinstance(data.ml_frame(stem, 5), pd.DataFrame)


def test_literals_reject_injection():
    assert data.date_lit(dt.date(2024, 1, 2)) == "DATE '2024-01-02'"
    with pytest.raises(TypeError):
        data.date_lit("2024-01-02; DROP TABLE x")
    assert data.str_lit("a'b") == "'a''b'"
    with pytest.raises(ValueError):
        data.table("gold; DROP")
    with pytest.raises(ValueError):
        data.hourly_activity_profile_sql("Nope", dt.date(2024, 1, 1), dt.date(2024, 2, 1))


# --------------------------------------------------------------------------- databricks portability


def all_sql() -> dict[str, str]:
    d0, d1 = dt.date(2025, 1, 1), dt.date(2025, 12, 31)
    out = {
        "date_bounds": data.date_bounds_sql(),
        "daily_summary": data.daily_summary_sql(d0, d1),
        "coverage": data.coverage_sql(),
        "weekly_summary": data.weekly_summary_sql(d0, d1),
        "hourly_heart_rate": data.hourly_heart_rate_profile_sql(d0, d1),
        "workouts": data.workouts_sql(d0, d1),
        "workout_hr_reference": data.workout_hr_reference_sql(),
        "route_points": data.route_points_sql("workout-routes/route_2024-05-01_10.00am.gpx"),
        "sleep": data.sleep_sql(d0, d1),
        "ecg_list": data.ecg_list_sql(),
        "ecg_samples": data.ecg_samples_sql("ecg_2022-12-25"),
        "ml_frame": data.ml_frame_sql("daily_anomalies", 10),
        "list_ml_tables": data.list_ml_tables_sql(),
    }
    for metric in data.HOURLY_METRICS:
        out[f"hourly_activity_{metric}"] = data.hourly_activity_profile_sql(metric, d0, d1)
    return out


def test_table_resolver(monkeypatch):
    assert data.table("gold_workouts") == "gold_workouts"
    assert data.ml_table("daily_anomalies") == "ml_daily_anomalies"
    monkeypatch.setenv("HEALTH_BACKEND", "databricks")
    assert data.table("gold_workouts") == "workspace.health.gold_workouts"
    assert data.ml_table("daily_anomalies") == "workspace.health.ml_daily_anomalies"
    monkeypatch.setenv("HEALTH_CATALOG", "health")
    monkeypatch.setenv("HEALTH_SCHEMA", "prod")
    assert data.table("silver_route_points") == "health.prod.silver_route_points"
    assert data.schema_name() == "health.prod"
    monkeypatch.setenv("HEALTH_SCHEMA", "bad-schema")
    with pytest.raises(ValueError):
        data.table("gold_workouts")


def test_sql_parses_as_databricks(monkeypatch):
    monkeypatch.setenv("HEALTH_BACKEND", "databricks")
    for name, sql in all_sql().items():
        if name == "list_ml_tables":
            assert sql == "SHOW TABLES IN workspace.health LIKE 'ml_*'"
            continue
        tree = sqlglot.parse_one(sql, read="databricks")
        tables = {t.sql(dialect="databricks") for t in tree.find_all(sqlglot.exp.Table)}
        ctes = {c.alias for c in tree.find_all(sqlglot.exp.CTE)}
        real = {t for t in tables if t.split(" ")[0] not in ctes}
        assert real, name
        for t in real:
            assert t.startswith("workspace.health."), (name, t)


@on_synthetic_data
def test_same_sql_runs_on_duckdb():
    """The local backend executes the very strings sent to Databricks (minus the prefix)."""
    for name, sql in all_sql().items():
        if name in ("list_ml_tables", "ml_frame"):
            continue
        sqlglot.parse_one(sql, read="duckdb")
        assert isinstance(data.run(sql), pd.DataFrame), name


# --------------------------------------------------------------------------- app


def app_test():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP / "app.py"), default_timeout=180)


def assert_clean(at, where: str):
    assert not at.exception, f"{where}: {[e.value for e in at.exception]}"
    assert not at.error, f"{where}: {[e.value for e in at.error]}"


@on_synthetic_data
@pytest.mark.parametrize("preset", ["Last 12 months", "Last 30 days", "All time"])
def test_every_page_renders(preset):
    at = app_test()
    at.run()
    assert_clean(at, "app")
    at.sidebar.selectbox(key="period_preset").select(preset).run()
    for page in PAGES:
        at.switch_page(page).run()
        assert_clean(at, f"{page} [{preset}]")
        assert at.title, page
        assert at.get("plotly_chart"), f"{page}: no charts rendered"


@on_synthetic_data
def test_interactive_widgets():
    at = app_test()
    at.run()
    at.switch_page("views/activity.py").run()
    for metric in list(data.HOURLY_METRICS)[1:3]:
        at.button_group[0].set_value(metric).run()
        assert_clean(at, f"activity metric {metric}")
    at.switch_page("views/workouts.py").run()
    routes = at.selectbox(key="w_route")
    for i in (1, 2):
        routes.select_index(i).run()
        assert_clean(at, "workout route")
    at.switch_page("views/ecg.py").run()
    ecg = at.selectbox(key="ecg_id")
    ecg.select_index(len(ecg.options) - 1).run()
    assert_clean(at, "ecg select")
    at.slider(key="ecg_t0").set_value(15.0).run()
    assert_clean(at, "ecg window")


def make_data_dir(tmp_path: Path, source: Path, ml: dict[str, pd.DataFrame]) -> Path:
    for layer in ("silver", "gold"):
        (tmp_path / layer).symlink_to(source / layer, target_is_directory=True)
    if ml:
        (tmp_path / "ml").mkdir()
        for stem, df in ml.items():
            df.to_parquet(tmp_path / "ml" / f"{stem}.parquet")
    return tmp_path


def test_ml_page_without_outputs(tmp_path, monkeypatch, synthetic_root):
    monkeypatch.setenv("HEALTH_DATA_DIR", str(make_data_dir(tmp_path, synthetic_root / "data", {})))
    at = app_test()
    at.run()
    at.switch_page("views/ml.py").run()
    assert_clean(at, "ml empty")
    assert any("No ML outputs found" in i.value for i in at.info)


def test_ml_page_with_synthetic_outputs(tmp_path, monkeypatch, synthetic_root):
    days = pd.date_range("2026-01-01", "2026-10-18", freq="D")
    n = len(days)
    rng = pd.Series(range(n))
    ml = {
        "daily_anomalies": pd.DataFrame(
            {
                "local_date": days.date,
                "anomaly_score": (rng % 17) / 17,
                "is_anomaly": rng % 40 == 0,
                "top_features": "steps +2.1sd",
            }
        ),
        "forecast_resting_hr": pd.concat(
            [
                pd.DataFrame({"local_date": days.date, "actual": 60 + rng % 5, "predicted": 61 + rng % 4, "model": m})
                for m in ("persistence", "lstm_28d")
            ]
        ),
        "forecast_metrics": pd.DataFrame(
            {
                "target": "resting_hr",
                "model": ["persistence", "lstm_28d"],
                "split": "test",
                "mae": [6.0, 5.0],
                "mae_skill_vs_persistence": [0.0, 0.17],
            }
        ),
        "day_clusters": pd.DataFrame(
            {
                "local_date": days.date,
                "cluster": rng % 3,
                "x": rng * 0.1,
                "y": rng % 7,
                "cluster_label": (rng % 3).map({0: "rest", 1: "walk", 2: "ride"}),
            }
        ),
        "similar_days": pd.DataFrame(
            {"local_date": days.date[:20], "neighbor_rank": 1, "neighbor_date": days.date[20:40], "distance": 1.0}
        ),
        "workout_clusters": pd.DataFrame(
            {
                "workout_id": ["a", "b"],
                "local_date": days.date[:2],
                "activity": "Cycling",
                "cluster": [0, 1],
                "x": [0.0, 1.0],
                "y": [1.0, 0.0],
            }
        ),
        "ecg_beat_anomalies": pd.DataFrame(
            {
                "ecg_id": "ecg_2022-12-25",
                "beat_index": range(30),
                "r_peak_s": [i * 0.8 for i in range(30)],
                "anomaly_score": [0.001 * i for i in range(30)],
                "is_anomaly": [i == 29 for i in range(30)],
            }
        ),
        "something_new": pd.DataFrame({"local_date": days.date[:10], "value": range(10)}),
    }
    monkeypatch.setenv("HEALTH_DATA_DIR", str(make_data_dir(tmp_path, synthetic_root / "data", ml)))
    at = app_test()
    at.run()
    at.switch_page("views/ml.py").run()
    assert_clean(at, "ml synthetic")
    labels = [t.label for t in at.tabs]
    for expected in (
        "Unusual days",
        "Forecasts",
        "Day types",
        "Similar days",
        "Workout clusters",
        "ECG beats",
        "Other tables",
    ):
        assert expected in labels, labels


# --------------------------------------------------------------------------- real server


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@on_synthetic_data
def test_server_health_endpoint():
    port = free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP / "app.py"),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
        ],
        cwd=APP,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        body = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
                    body = r.read().decode()
                    break
            except OSError:
                time.sleep(0.5)
        assert body == "ok"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            assert r.status == 200
    finally:
        proc.terminate()
        proc.wait(timeout=20)
