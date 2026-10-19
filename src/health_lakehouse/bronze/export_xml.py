"""Stream-parse Apple Health `export.xml` into raw bronze Parquet files.

The export is a single multi-GB XML document, so it is parsed with
`lxml.etree.iterparse` and elements are freed as soon as they are written.
Memory stays flat regardless of export size, which also makes this safe to
run on a single Databricks driver.

Bronze keeps every attribute as a raw string. Typing, timezone handling,
deduplication and unit harmonisation happen in silver.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from lxml import etree

from .writer import BufferedParquetWriter, string_schema

INGEST_COLS = ("_source_file", "_ingested_at")

RECORD_SCHEMA = string_schema(
    "record_id",
    "type",
    "source_name",
    "source_version",
    "device",
    "unit",
    "value",
    "creation_date",
    "start_date",
    "end_date",
    "metadata_json",
    "correlation_id",
    *INGEST_COLS,
)
HRV_BPM_SCHEMA = string_schema("record_id", "bpm", "time", *INGEST_COLS)
CORRELATION_SCHEMA = string_schema(
    "correlation_id",
    "type",
    "source_name",
    "source_version",
    "device",
    "creation_date",
    "start_date",
    "end_date",
    "metadata_json",
    *INGEST_COLS,
)
WORKOUT_SCHEMA = string_schema(
    "workout_id",
    "workout_activity_type",
    "duration",
    "duration_unit",
    "total_distance",
    "total_distance_unit",
    "total_energy_burned",
    "total_energy_burned_unit",
    "source_name",
    "source_version",
    "device",
    "creation_date",
    "start_date",
    "end_date",
    "metadata_json",
    "route_file",
    *INGEST_COLS,
)
WORKOUT_EVENT_SCHEMA = string_schema(
    "workout_id",
    "type",
    "date",
    "duration",
    "duration_unit",
    "metadata_json",
    *INGEST_COLS,
)
WORKOUT_STAT_SCHEMA = string_schema(
    "workout_id",
    "type",
    "start_date",
    "end_date",
    "average",
    "minimum",
    "maximum",
    "sum",
    "unit",
    *INGEST_COLS,
)
ACTIVITY_SCHEMA = string_schema(
    "date_components",
    "active_energy_burned",
    "active_energy_burned_goal",
    "active_energy_burned_unit",
    "apple_move_time",
    "apple_move_time_goal",
    "apple_exercise_time",
    "apple_exercise_time_goal",
    "apple_stand_hours",
    "apple_stand_hours_goal",
    *INGEST_COLS,
)
ME_SCHEMA = string_schema("attribute", "value", *INGEST_COLS)

# XML attribute -> snake_case column, for the generic copy helpers below.
_ATTR_MAP = {
    "type": "type",
    "sourceName": "source_name",
    "sourceVersion": "source_version",
    "device": "device",
    "unit": "unit",
    "value": "value",
    "creationDate": "creation_date",
    "startDate": "start_date",
    "endDate": "end_date",
    "workoutActivityType": "workout_activity_type",
    "duration": "duration",
    "durationUnit": "duration_unit",
    "totalDistance": "total_distance",
    "totalDistanceUnit": "total_distance_unit",
    "totalEnergyBurned": "total_energy_burned",
    "totalEnergyBurnedUnit": "total_energy_burned_unit",
    "date": "date",
    "average": "average",
    "minimum": "minimum",
    "maximum": "maximum",
    "sum": "sum",
    "dateComponents": "date_components",
    "activeEnergyBurned": "active_energy_burned",
    "activeEnergyBurnedGoal": "active_energy_burned_goal",
    "activeEnergyBurnedUnit": "active_energy_burned_unit",
    "appleMoveTime": "apple_move_time",
    "appleMoveTimeGoal": "apple_move_time_goal",
    "appleExerciseTime": "apple_exercise_time",
    "appleExerciseTimeGoal": "apple_exercise_time_goal",
    "appleStandHours": "apple_stand_hours",
    "appleStandHoursGoal": "apple_stand_hours_goal",
}


def _attrs(elem) -> dict:
    return {_ATTR_MAP.get(k, k): v for k, v in elem.attrib.items()}


def _stable_id(*parts: str | None) -> str:
    return hashlib.sha1("|".join(p or "" for p in parts).encode()).hexdigest()


def _metadata(elem) -> str | None:
    meta = {m.get("key"): m.get("value") for m in elem.iterfind("MetadataEntry")}
    return json.dumps(meta, ensure_ascii=False) if meta else None


class _Writers:
    def __init__(self, out_dir: Path):
        self.tables = {
            "records": BufferedParquetWriter(out_dir / "records.parquet", RECORD_SCHEMA),
            "hrv_instantaneous_bpm": BufferedParquetWriter(out_dir / "hrv_instantaneous_bpm.parquet", HRV_BPM_SCHEMA),
            "correlations": BufferedParquetWriter(out_dir / "correlations.parquet", CORRELATION_SCHEMA),
            "workouts": BufferedParquetWriter(out_dir / "workouts.parquet", WORKOUT_SCHEMA),
            "workout_events": BufferedParquetWriter(out_dir / "workout_events.parquet", WORKOUT_EVENT_SCHEMA),
            "workout_statistics": BufferedParquetWriter(out_dir / "workout_statistics.parquet", WORKOUT_STAT_SCHEMA),
            "activity_summaries": BufferedParquetWriter(out_dir / "activity_summaries.parquet", ACTIVITY_SCHEMA),
            "me": BufferedParquetWriter(out_dir / "me.parquet", ME_SCHEMA),
        }

    def __getitem__(self, key: str) -> BufferedParquetWriter:
        return self.tables[key]

    def close(self) -> dict[str, int]:
        return {name: w.close() for name, w in self.tables.items()}


def _record_row(elem, ingest: dict, correlation_id: str | None = None) -> dict:
    row = _attrs(elem)
    row["record_id"] = _stable_id(
        row.get("type"),
        row.get("source_name"),
        row.get("start_date"),
        row.get("end_date"),
        row.get("value"),
        row.get("creation_date"),
    )
    row["metadata_json"] = _metadata(elem)
    row["correlation_id"] = correlation_id
    row.update(ingest)
    return row


def _write_record(elem, w: _Writers, ingest: dict, correlation_id: str | None = None) -> None:
    row = _record_row(elem, ingest, correlation_id)
    w["records"].write(row)
    hrv = elem.find("HeartRateVariabilityMetadataList")
    if hrv is not None:
        for beat in hrv.iterfind("InstantaneousBeatsPerMinute"):
            w["hrv_instantaneous_bpm"].write(
                {"record_id": row["record_id"], "bpm": beat.get("bpm"), "time": beat.get("time"), **ingest}
            )


def _write_workout(elem, w: _Writers, ingest: dict) -> None:
    row = _attrs(elem)
    workout_id = _stable_id(
        row.get("workout_activity_type"),
        row.get("source_name"),
        row.get("start_date"),
        row.get("end_date"),
    )
    route = elem.find("WorkoutRoute/FileReference")
    row.update(
        workout_id=workout_id,
        metadata_json=_metadata(elem),
        route_file=route.get("path") if route is not None else None,
        **ingest,
    )
    w["workouts"].write(row)
    for ev in elem.iterfind("WorkoutEvent"):
        w["workout_events"].write({**_attrs(ev), "workout_id": workout_id, "metadata_json": _metadata(ev), **ingest})
    for st in elem.iterfind("WorkoutStatistics"):
        w["workout_statistics"].write({**_attrs(st), "workout_id": workout_id, **ingest})


def parse_export_xml(xml_path: str | Path, out_dir: str | Path, progress_every: int = 1_000_000) -> dict[str, int]:
    """Parse `export.xml` into one Parquet file per bronze table in `out_dir`.

    Returns the number of rows written per table.
    """
    xml_path, out_dir = Path(xml_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ingest = {"_source_file": xml_path.name, "_ingested_at": datetime.now(UTC).isoformat()}
    w = _Writers(out_dir)

    started = time.time()
    seen = 0
    context = etree.iterparse(
        str(xml_path), events=("end",), huge_tree=True, recover=True, load_dtd=False, resolve_entities=False
    )
    for _, elem in context:
        tag = elem.tag
        parent = elem.getparent()
        top_level = parent is not None and parent.getparent() is None

        if tag == "Record" and top_level:
            _write_record(elem, w, ingest)
        elif tag == "Correlation":
            row = _attrs(elem)
            cid = _stable_id(row.get("type"), row.get("source_name"), row.get("start_date"), row.get("end_date"))
            w["correlations"].write({**row, "correlation_id": cid, "metadata_json": _metadata(elem), **ingest})
            for rec in elem.iterfind("Record"):
                _write_record(rec, w, ingest, correlation_id=cid)
        elif tag == "Workout":
            _write_workout(elem, w, ingest)
        elif tag == "ActivitySummary":
            w["activity_summaries"].write({**_attrs(elem), **ingest})
        elif tag == "Me":
            for k, v in elem.attrib.items():
                w["me"].write({"attribute": k, "value": v, **ingest})
        elif not top_level:
            # Nested children (MetadataEntry, WorkoutEvent, ...) are consumed
            # by their parent handler; keep them until the parent ends.
            continue

        # Free memory: clear the finished top-level element and drop siblings
        # that were already processed.
        if top_level:
            elem.clear()
            while elem.getprevious() is not None:
                del parent[0]
            seen += 1
            if progress_every and seen % progress_every == 0:
                print(f"  {seen:,} top-level elements in {time.time() - started:,.0f}s", flush=True)

    counts = w.close()
    print(f"export.xml parsed in {time.time() - started:,.0f}s: {counts}", flush=True)
    return counts
