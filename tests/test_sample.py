"""The synthetic export generator: structure, determinism, locales and the signals it injects."""

from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest
from conftest import SYNTH_DAYS, SYNTH_SEED
from lxml import etree

from health_lakehouse.sample import DEFAULT_END, _daily_state, generate

EXPECTED_TYPES = {
    "HKQuantityTypeIdentifierStepCount",
    "HKQuantityTypeIdentifierDistanceWalkingRunning",
    "HKQuantityTypeIdentifierActiveEnergyBurned",
    "HKQuantityTypeIdentifierBasalEnergyBurned",
    "HKQuantityTypeIdentifierFlightsClimbed",
    "HKQuantityTypeIdentifierAppleExerciseTime",
    "HKQuantityTypeIdentifierAppleStandTime",
    "HKCategoryTypeIdentifierAppleStandHour",
    "HKQuantityTypeIdentifierHeartRate",
    "HKQuantityTypeIdentifierRestingHeartRate",
    "HKQuantityTypeIdentifierWalkingHeartRateAverage",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "HKQuantityTypeIdentifierOxygenSaturation",
    "HKQuantityTypeIdentifierRespiratoryRate",
    "HKQuantityTypeIdentifierVO2Max",
    "HKQuantityTypeIdentifierBodyMass",
    "HKQuantityTypeIdentifierBodyMassIndex",
    "HKQuantityTypeIdentifierWalkingSpeed",
    "HKQuantityTypeIdentifierWalkingStepLength",
    "HKQuantityTypeIdentifierWalkingAsymmetryPercentage",
    "HKQuantityTypeIdentifierWalkingDoubleSupportPercentage",
    "HKQuantityTypeIdentifierEnvironmentalAudioExposure",
    "HKCategoryTypeIdentifierSleepAnalysis",
    "HKQuantityTypeIdentifierBloodPressureSystolic",
    "HKQuantityTypeIdentifierBloodPressureDiastolic",
    "HKCategoryTypeIdentifierMindfulSession",
    "HKDataTypeSleepDurationGoal",
}


def q(sql: str):
    return duckdb.sql(sql).fetchall()


def digest(root) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_export_xml_is_valid_and_shaped_like_apples(synthetic_export):
    tree = etree.parse(str(synthetic_export / "export.xml"), etree.XMLParser(huge_tree=True, load_dtd=True))
    root = tree.getroot()
    assert root.tag == "HealthData" and root.get("locale") == "en_DE"
    assert tree.docinfo.internalDTD is not None
    assert [c.tag for c in root[:2]] == ["ExportDate", "Me"]
    tags = {c.tag for c in root}
    assert {"Record", "Correlation", "Workout", "ActivitySummary", "ClinicalRecord"} <= tags
    assert tree.docinfo.internalDTD.validate(root), tree.docinfo.internalDTD.error_log
    # Apple stamps every date with the offset active at export time.
    text = (synthetic_export / "export.xml").read_text(encoding="utf-8")
    offsets = set(
        re.findall(r'(?:startDate|endDate|creationDate|date)="\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ([+-]\d{4})"', text)
    )
    assert offsets == {"+0200"}


def test_bronze_has_every_expected_type_and_both_sources(synthetic_bronze):
    rec = (synthetic_bronze / "records.parquet").as_posix()
    types = {t for (t,) in q(f"SELECT DISTINCT type FROM '{rec}'")}
    assert types >= EXPECTED_TYPES, EXPECTED_TYPES - types
    sources = {s for (s,) in q(f"SELECT DISTINCT source_name FROM '{rec}' WHERE type LIKE '%StepCount'")}
    assert sources == {"Jane's Apple Watch", "Jane's iPhone"}
    (n_meta,) = q(f"SELECT count(*) FROM '{rec}' WHERE metadata_json LIKE '%HeartRateMotionContext%'")[0]
    assert n_meta > 1000
    hrv = (synthetic_bronze / "hrv_instantaneous_bpm.parquet").as_posix()
    assert q(f"SELECT count(DISTINCT record_id) > 100 FROM '{hrv}'")[0][0]
    w = (synthetic_bronze / "workouts.parquet").as_posix()
    acts = {a for (a,) in q(f"SELECT DISTINCT workout_activity_type FROM '{w}'")}
    assert {
        "HKWorkoutActivityTypeCycling",
        "HKWorkoutActivityTypeRunning",
        "HKWorkoutActivityTypeSwimming",
        "HKWorkoutActivityTypeYoga",
        "HKWorkoutActivityTypeTraditionalStrengthTraining",
    } <= acts
    for key in (
        "HKIndoorWorkout",
        "HKTimeZone",
        "HKWeatherTemperature",
        "HKWeatherHumidity",
        "HKAverageMETs",
        "HKElevationAscended",
    ):
        assert q(f"SELECT count(*) FROM '{w}' WHERE metadata_json LIKE '%\"{key}\"%'")[0][0] > 0, key
    ev = (synthetic_bronze / "workout_events.parquet").as_posix()
    assert {
        "HKWorkoutEventTypePause",
        "HKWorkoutEventTypeResume",
        "HKWorkoutEventTypeLap",
        "HKWorkoutEventTypeSegment",
    } <= {t for (t,) in q(f"SELECT DISTINCT type FROM '{ev}'")}


def test_routes_match_file_references(synthetic_export, synthetic_bronze):
    w = (synthetic_bronze / "workouts.parquet").as_posix()
    refs = {r for (r,) in q(f"SELECT route_file FROM '{w}' WHERE route_file IS NOT NULL")}
    files = {f"/workout-routes/{p.name}" for p in (synthetic_export / "workout-routes").glob("*.gpx")}
    assert refs and refs == files
    pts = (synthetic_bronze / "route_points.parquet").as_posix()
    assert {r for (r,) in q(f"SELECT DISTINCT route_file FROM '{pts}'")} == refs
    assert q(f"SELECT count(*) FROM '{pts}' WHERE speed IS NULL OR h_acc IS NULL OR ele IS NULL")[0][0] == 0


def test_ecgs_in_both_locales_parse_in_silver(synthetic_export, synthetic_root):
    csvs = sorted((synthetic_export / "electrocardiograms").glob("*.csv"))
    texts = [p.read_text(encoding="utf-8") for p in csvs]
    assert len(csvs) == 6
    assert any("Sample Rate,512,000 hertz" in t and "\n-" in t and re.search(r"^-?\d+,\d{3}$", t, re.M) for t in texts)
    assert any("Sample Rate,512.000 hertz" in t and re.search(r"^-?\d+\.\d{3}$", t, re.M) for t in texts)
    assert all(t.startswith("Name,Jane Appleseed\n") for t in texts)
    gold = pd.read_parquet(synthetic_root / "data" / "gold" / "gold_ecg_summary.parquet")
    assert (gold["sample_rate_hz"] == 512.0).all()
    assert (gold["duration_s"] == 30.0).all()
    assert {"Sinus Rhythm", "Atrial Fibrillation", "Inconclusive"} == set(gold["classification"])
    samples = pd.read_parquet(synthetic_root / "data" / "silver" / "silver_ecg_samples.parquet")
    assert samples["value_uv"].notna().all() and samples["value_uv"].max() > 800


def test_clinical_records(synthetic_root):
    c = pd.read_parquet(synthetic_root / "data" / "silver" / "silver_clinical_records.parquet")
    assert set(c["resource_type"]) == {"Immunization", "Observation"}
    imm = c[c["resource_type"] == "Immunization"]
    assert imm["vaccine"].notna().all() and imm["occurrence_date"].notna().all()


def test_deterministic_per_seed(tmp_path):
    generate(tmp_path / "a", days=20, seed=1)
    generate(tmp_path / "b", days=20, seed=1)
    generate(tmp_path / "c", days=20, seed=2)
    assert digest(tmp_path / "a") == digest(tmp_path / "b")
    assert digest(tmp_path / "a") != digest(tmp_path / "c")


def test_other_timezone_and_offset(tmp_path):
    generate(tmp_path / "ny", days=10, seed=3, tz="America/New_York", end=date(2026, 1, 15))
    text = (tmp_path / "ny" / "export.xml").read_text(encoding="utf-8")
    assert '<HealthData locale="en_US">' in text
    assert set(re.findall(r'startDate="[^"]+ ([+-]\d{4})"', text)) == {"-0500"}
    assert '"HKTimeZone" value="America/New_York"' in text or "HKTimeZone" not in text


@pytest.fixture(scope="module")
def daily(synthetic_root):
    d = pd.read_parquet(synthetic_root / "data" / "gold" / "gold_daily_summary.parquet")
    d["local_date"] = pd.to_datetime(d["local_date"])
    return d.set_index("local_date").sort_index()


def test_local_times_follow_dst(synthetic_root):
    """Resting HR samples start at 00:05 local on every day, on both sides of the DST switches."""
    s = pd.read_parquet(
        synthetic_root / "data" / "silver" / "silver_records.parquet", columns=["metric", "start_local", "local_date"]
    )
    rhr = s[s["metric"] == "RestingHeartRate"]
    assert (rhr["start_local"].dt.strftime("%H:%M") == "00:05").all()
    months = set(pd.to_datetime(rhr["local_date"]).dt.month)
    assert {1, 7} <= months  # winter (CET) and summer (CEST)


def test_dedup_has_overlapping_sources(synthetic_root):
    h = pd.read_parquet(synthetic_root / "data" / "gold" / "gold_hourly_activity.parquet")
    steps = h[h["metric"] == "StepCount"]
    assert (steps["n_sources"] == 2).mean() > 0.5
    assert set(steps["chosen_source"]) == {"Jane's Apple Watch", "Jane's iPhone"}


def test_weekly_rhythm_trend_and_illness(daily):
    weekday = daily[daily["day_of_week"].between(2, 6)]
    weekend = daily[daily["day_of_week"].isin([1, 7])]
    assert weekday["steps"].mean() > weekend["steps"].mean()
    assert weekend["workout_min"].mean() > weekday["workout_min"].mean()
    rhr_by_dow = daily.groupby("day_of_week")["resting_hr"].mean()
    assert rhr_by_dow[[2, 3]].mean() > rhr_by_dow[[1, 7]].mean()  # Mon/Tue above the weekend
    first, last = daily.iloc[:90], daily.iloc[-90:]
    assert last["resting_hr"].mean() < first["resting_hr"].mean()
    assert last["hrv_sdnn_ms"].mean() > first["hrv_sdnn_ms"].mean()

    states = _daily_state(np.random.default_rng(SYNTH_SEED), DEFAULT_END - timedelta(days=SYNTH_DAYS - 1), SYNTH_DAYS)
    ill = pd.to_datetime([s.day for s in states if s.illness > 0.5])
    off = pd.to_datetime([s.day for s in states if not s.watch])
    assert len(ill) >= 2 and len(off) >= 3
    base = daily.loc[~daily.index.isin(ill), "resting_hr"].median()
    assert daily.loc[ill, "resting_hr"].mean() > base + 4
    assert daily.loc[ill, "steps"].mean() < 0.6 * daily["steps"].median()
    assert daily.loc[daily.index.isin(off), "resting_hr"].isna().all()  # watch not worn: no RHR
    assert daily.loc[daily.index.isin(off), "steps"].notna().all()  # but the phone still counts
    assert daily["wrist_temp_c"].isna().iloc[:100].all() and daily["wrist_temp_c"].notna().iloc[-60:].any()


def test_size_and_speed(tmp_path):
    import time

    t0 = time.time()
    generate(tmp_path / "x", days=60, seed=5)
    assert time.time() - t0 < 10
    size = sum(p.stat().st_size for p in (tmp_path / "x").rglob("*") if p.is_file())
    assert size < 15e6
