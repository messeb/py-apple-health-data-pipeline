"""Bronze parser tests on a tiny synthetic Apple Health export."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from health_lakehouse.bronze import build_bronze

EXPORT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData [
<!ELEMENT HealthData (ExportDate,Me,(Record|Correlation|Workout|ActivitySummary)*)>
]>
<HealthData locale="de_DE">
 <ExportDate value="2026-10-18 13:00:00 +0200"/>
 <Me HKCharacteristicTypeIdentifierDateOfBirth="1990-01-01" HKCharacteristicTypeIdentifierBiologicalSex="HKBiologicalSexFemale"/>
 <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Watch" unit="count" creationDate="2024-01-15 09:10:00 +0200" startDate="2024-01-15 09:00:00 +0200" endDate="2024-01-15 09:05:00 +0200" value="120"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Watch" unit="count/min" creationDate="2024-01-15 09:10:00 +0200" startDate="2024-01-15 09:00:00 +0200" endDate="2024-01-15 09:00:00 +0200" value="61">
  <MetadataEntry key="HKMetadataKeyHeartRateMotionContext" value="1"/>
 </Record>
 <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN" sourceName="Watch" unit="ms" creationDate="2024-01-15 09:10:00 +0200" startDate="2024-01-15 09:00:00 +0200" endDate="2024-01-15 09:01:00 +0200" value="42.5">
  <HeartRateVariabilityMetadataList>
   <InstantaneousBeatsPerMinute bpm="60" time="9:00:01,00"/>
   <InstantaneousBeatsPerMinute bpm="62" time="9:00:02,00"/>
  </HeartRateVariabilityMetadataList>
 </Record>
 <Correlation type="HKCorrelationTypeIdentifierBloodPressure" sourceName="Cuff" creationDate="2024-01-15 10:00:00 +0200" startDate="2024-01-15 10:00:00 +0200" endDate="2024-01-15 10:00:00 +0200">
  <Record type="HKQuantityTypeIdentifierBloodPressureSystolic" sourceName="Cuff" unit="mmHg" startDate="2024-01-15 10:00:00 +0200" endDate="2024-01-15 10:00:00 +0200" value="120"/>
  <Record type="HKQuantityTypeIdentifierBloodPressureDiastolic" sourceName="Cuff" unit="mmHg" startDate="2024-01-15 10:00:00 +0200" endDate="2024-01-15 10:00:00 +0200" value="80"/>
 </Correlation>
 <Workout workoutActivityType="HKWorkoutActivityTypeCycling" duration="30" durationUnit="min" sourceName="Watch" startDate="2024-01-15 17:00:00 +0200" endDate="2024-01-15 17:30:00 +0200">
  <MetadataEntry key="HKIndoorWorkout" value="0"/>
  <WorkoutEvent type="HKWorkoutEventTypePause" date="2024-01-15 17:10:00 +0200"/>
  <WorkoutStatistics type="HKQuantityTypeIdentifierActiveEnergyBurned" startDate="2024-01-15 17:00:00 +0200" endDate="2024-01-15 17:30:00 +0200" sum="250" unit="kcal"/>
  <WorkoutRoute sourceName="Watch" startDate="2024-01-15 17:00:00 +0200" endDate="2024-01-15 17:30:00 +0200">
   <FileReference path="/workout-routes/route_1.gpx"/>
  </WorkoutRoute>
 </Workout>
 <ActivitySummary dateComponents="2024-01-15" activeEnergyBurned="500" activeEnergyBurnedGoal="400" activeEnergyBurnedUnit="kcal" appleExerciseTime="30" appleExerciseTimeGoal="30" appleStandHours="10" appleStandHoursGoal="12"/>
</HealthData>
"""

GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
<trkpt lon="-0.1" lat="51.5"><ele>46.8</ele><time>2024-01-15T16:00:00Z</time><extensions><speed>5.1</speed><course>10</course><hAcc>3</hAcc><vAcc>2</vAcc></extensions></trkpt>
<trkpt lon="-0.2" lat="51.4"><ele>47.0</ele><time>2024-01-15T16:00:01Z</time><extensions><speed>5.2</speed></extensions></trkpt>
</trkseg></trk></gpx>
"""

ECG = """Name,Jane Doe
Date of Birth,"1. Jan 1990"
Recorded Date,2024-01-15 12:00:00 +0200
Classification,Sinus Rhythm
Symptoms,
Software Version,1.90
Device,"Watch7,1"
Sample Rate,512 hertz


Lead,Lead I
Unit,µV

-1,195
2,5
"""


@pytest.fixture(scope="module")
def bronze(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    export = root / "apple_health_export"
    (export / "workout-routes").mkdir(parents=True)
    (export / "electrocardiograms").mkdir()
    (export / "clinical-records").mkdir()
    (export / "export.xml").write_text(EXPORT_XML, encoding="utf-8")
    (export / "workout-routes" / "route_1.gpx").write_text(GPX, encoding="utf-8")
    (export / "electrocardiograms" / "ecg_2024-01-15.csv").write_text(ECG, encoding="utf-8")
    (export / "clinical-records" / "Immunization-1.json").write_text(
        json.dumps({"resourceType": "Immunization", "status": "completed"}), encoding="utf-8"
    )
    out = root / "bronze"
    counts = build_bronze(export, out)
    return out, counts


def read(bronze, table):
    return pd.read_parquet(bronze[0] / f"{table}.parquet")


def test_counts(bronze):
    _, counts = bronze
    assert counts["records"] == 5  # 3 top-level + 2 inside the correlation
    assert counts["correlations"] == 1
    assert counts["hrv_instantaneous_bpm"] == 2
    assert counts["workouts"] == 1
    assert counts["workout_events"] == 1
    assert counts["workout_statistics"] == 1
    assert counts["activity_summaries"] == 1
    assert counts["route_points"] == 2
    assert counts["ecg_readings"] == 1
    assert counts["ecg_samples"] == 2
    assert counts["clinical_records"] == 1


def test_records_keep_raw_strings_and_metadata(bronze):
    rec = read(bronze, "records")
    hr = rec[rec.type == "HKQuantityTypeIdentifierHeartRate"].iloc[0]
    assert hr.value == "61"
    assert hr.start_date == "2024-01-15 09:00:00 +0200"
    assert json.loads(hr.metadata_json) == {"HKMetadataKeyHeartRateMotionContext": "1"}
    assert rec.record_id.is_unique


def test_correlation_children_are_linked(bronze):
    rec = read(bronze, "records")
    corr = read(bronze, "correlations")
    bp = rec[rec.type.str.startswith("HKQuantityTypeIdentifierBloodPressure")]
    assert len(bp) == 2
    assert set(bp.correlation_id) == set(corr.correlation_id)


def test_workout_children_share_workout_id(bronze):
    w = read(bronze, "workouts").iloc[0]
    assert w.route_file == "/workout-routes/route_1.gpx"
    assert read(bronze, "workout_statistics").workout_id.iloc[0] == w.workout_id
    assert read(bronze, "workout_events").workout_id.iloc[0] == w.workout_id


def test_route_file_matches_workout_reference(bronze):
    pts = read(bronze, "route_points")
    assert set(pts.route_file) == {"/workout-routes/route_1.gpx"}
    assert pts.speed.tolist() == ["5.1", "5.2"]


def test_ecg_drops_personal_data(bronze):
    header = json.loads(read(bronze, "ecg_readings").header_json.iloc[0])
    assert "Name" not in header and "Date of Birth" not in header
    assert header["Classification"] == "Sinus Rhythm"
    assert read(bronze, "ecg_samples").value_raw.tolist() == ["-1,195", "2,5"]


def test_silver_gold_build_on_synthetic_bronze(bronze, tmp_path):
    """The full SQL pipeline runs end to end on the synthetic bronze data."""
    from health_lakehouse.pipeline import build_layers

    counts = build_layers(tmp_path / "t.duckdb", bronze[0])
    assert counts["silver_records"] == 5
    assert counts["gold_daily_summary"] == 1

    import duckdb

    con = duckdb.connect(str(tmp_path / "t.duckdb"), read_only=True)
    steps, rhr_hour = con.execute(
        "SELECT a.steps, (SELECT local_hour FROM silver_records WHERE metric='StepCount') FROM gold_daily_activity a"
    ).fetchone()
    assert steps == 120
    # 09:00 +0200 in January = 07:00 UTC = 08:00 Europe/Berlin (CET)
    assert rhr_hour == 8
    sys_bp = con.execute("SELECT bp_systolic FROM gold_daily_vitals").fetchone()[0]
    assert sys_bp == 120
