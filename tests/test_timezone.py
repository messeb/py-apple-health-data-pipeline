"""`--timezone auto`: detection from HKTimeZone metadata, fallbacks, and the CLI default."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from health_lakehouse import timezone as tzmod
from health_lakehouse.cli import main
from health_lakehouse.sample import generate
from health_lakehouse.sql_compat import render
from health_lakehouse.timezone import TIMEZONE_COUNTS_SQL, is_iana, pick_timezone, resolve_timezone


@pytest.mark.parametrize(
    "name, ok",
    [
        ("Europe/Berlin", True),
        ("America/New_York", True),
        ("UTC", True),
        ("CET", False),
        ("EST", False),
        ("Mars/Olympus", False),
        ("", False),
        (None, False),
        (" Europe/Berlin", False),
    ],
)
def test_is_iana(name, ok):
    assert is_iana(name) is ok


def test_pick_most_common_valid_zone():
    rows = [("Europe/Lisbon", 7), ("Europe/Berlin", 30), ("CET", 99), (None, 5), ("Europe/Berlin", 2)]
    assert pick_timezone(rows) == ("Europe/Berlin", 32, 39)
    assert pick_timezone([("Asia/Tokyo", 3), ("Africa/Cairo", 3)])[0] == "Africa/Cairo"  # ties: alphabetical
    assert pick_timezone([("CET", 3)]) is None


def write_bronze(dir_, workouts: list[dict | None], records: list[dict | None]):
    dir_.mkdir(parents=True, exist_ok=True)
    for name, metas in (("workouts", workouts), ("records", records)):
        col = [json.dumps(m) if m is not None else None for m in metas]
        pq.write_table(pa.table({"metadata_json": pa.array(col, pa.string())}), dir_ / f"{name}.parquet")


def test_detect_from_bronze(tmp_path):
    write_bronze(
        tmp_path,
        [{"HKTimeZone": "America/Chicago"}] * 3 + [{"HKIndoorWorkout": "1"}, None],
        [{"HKTimeZone": "America/Chicago"}, {"HKTimeZone": "Europe/Paris"}, None],
    )
    assert tzmod.detect_from_bronze(tmp_path) == ("America/Chicago", 4, 5)
    assert resolve_timezone("auto", tmp_path, log=lambda *_: None) == "America/Chicago"


def test_detect_without_metadata_falls_back(tmp_path, monkeypatch):
    write_bronze(tmp_path, [None, {"HKIndoorWorkout": "0"}], [None])
    assert tzmod.detect_from_bronze(tmp_path) is None
    assert tzmod.detect_from_bronze(tmp_path / "missing") is None
    monkeypatch.setattr(tzmod, "system_timezone", lambda: "Asia/Tokyo")
    logs = []
    assert resolve_timezone("auto", tmp_path, log=logs.append) == "Asia/Tokyo"
    assert "machine" in logs[0]
    monkeypatch.setattr(tzmod, "system_timezone", lambda: None)
    assert resolve_timezone("auto", tmp_path, log=logs.append) == "UTC"


def test_explicit_zone_is_kept_and_validated(tmp_path):
    assert resolve_timezone("Europe/Berlin", tmp_path) == "Europe/Berlin"
    assert resolve_timezone("AUTO", None, log=lambda *_: None)  # case-insensitive keyword
    with pytest.raises(ValueError):
        resolve_timezone("Berlin", tmp_path)


def test_system_timezone_reads_tz_env(monkeypatch):
    monkeypatch.setenv("TZ", "Australia/Sydney")
    assert tzmod.system_timezone() == "Australia/Sydney"
    monkeypatch.setenv("TZ", ":Pacific/Auckland")
    assert tzmod.system_timezone() == "Pacific/Auckland"
    monkeypatch.setenv("TZ", "bogus")
    assert tzmod.system_timezone() is None or is_iana(tzmod.system_timezone())


def test_counts_sql_is_databricks_sql():
    import sqlglot

    sqlglot.parse_one(render(TIMEZONE_COUNTS_SQL, {}), read="databricks")


def test_synthetic_export_detects_its_zone(synthetic_bronze):
    assert tzmod.detect_from_bronze(synthetic_bronze)[0] == "Europe/Berlin"


def test_cli_auto_end_to_end_new_york(tmp_path, capsys):
    """A New York export (stamped -0500 in January) gets New York local times without --timezone."""
    export, data = tmp_path / "export", tmp_path / "data"
    generate(export, days=14, seed=11, tz="America/New_York", end=date(2026, 1, 20))
    main(["all", "--export-dir", str(export), "--data-dir", str(data)])
    assert "timezone: America/New_York (auto" in capsys.readouterr().out
    s = pd.read_parquet(data / "silver" / "silver_records.parquet", columns=["metric", "start_local"])
    rhr = s[s["metric"] == "RestingHeartRate"]
    assert not rhr.empty and (rhr["start_local"].dt.strftime("%H:%M") == "00:05").all()

    main(["silver", "--export-dir", str(export), "--data-dir", str(data), "--timezone", "UTC"])
    s = pd.read_parquet(data / "silver" / "silver_records.parquet", columns=["metric", "start_local"])
    assert (s[s["metric"] == "RestingHeartRate"]["start_local"].dt.strftime("%H:%M") == "05:05").all()
