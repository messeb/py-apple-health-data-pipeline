"""Bronze parsers for the side files of an Apple Health export.

* `electrocardiograms/*.csv`  - Apple Watch ECG recordings (header + Lead I samples)
* `workout-routes/*.gpx`      - GPS tracks referenced by workouts
* `clinical-records/*.json`   - FHIR resources (immunizations, labs, ...)

ECG CSVs are written with the phone's locale, so a German export uses a
decimal comma (`-1,195` = -1.195 µV). Bronze keeps the raw string; silver
converts it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
from lxml import etree

from .writer import BufferedParquetWriter, string_schema

INGEST_COLS = ("_source_file", "_ingested_at")

ECG_READING_SCHEMA = string_schema("ecg_id", "header_json", *INGEST_COLS)
ECG_SAMPLE_SCHEMA = string_schema("ecg_id", "value_raw", *INGEST_COLS, extra={"sample_index": pa.int32()})
ROUTE_POINT_SCHEMA = string_schema(
    "route_file",
    "lon",
    "lat",
    "ele",
    "time",
    "speed",
    "course",
    "h_acc",
    "v_acc",
    *INGEST_COLS,
    extra={"point_index": pa.int32()},
)
CLINICAL_SCHEMA = string_schema("record_file", "resource_type", "raw_json", *INGEST_COLS)

_GPX_NS = "{http://www.topografix.com/GPX/1/1}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def parse_ecgs(ecg_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    ecg_dir, out_dir = Path(ecg_dir), Path(out_dir)
    readings = BufferedParquetWriter(out_dir / "ecg_readings.parquet", ECG_READING_SCHEMA)
    samples = BufferedParquetWriter(out_dir / "ecg_samples.parquet", ECG_SAMPLE_SCHEMA)
    ingested_at = _now()
    for path in sorted(ecg_dir.glob("*.csv")):
        ingest = {"_source_file": path.name, "_ingested_at": ingested_at}
        ecg_id = path.stem
        header: dict[str, str] = {}
        in_samples = False
        idx = 0
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not in_samples:
                if not line.strip():
                    continue
                key, _, value = line.partition(",")
                value = value.strip().strip('"')
                # The last header line is the unit; samples follow.
                header[key.strip()] = value
                if key.strip() == "Unit":
                    in_samples = True
                continue
            if not line.strip():
                continue
            samples.write({"ecg_id": ecg_id, "sample_index": idx, "value_raw": line.strip().strip('"'), **ingest})
            idx += 1
        # The person's name and birth date sit in every ECG header - drop them
        # at the earliest layer; they are not needed for analytics.
        header.pop("Name", None)
        header.pop("Date of Birth", None)
        header["n_samples"] = str(idx)
        readings.write({"ecg_id": ecg_id, "header_json": json.dumps(header, ensure_ascii=False), **ingest})
    return {"ecg_readings": readings.close(), "ecg_samples": samples.close()}


def parse_routes(routes_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    routes_dir, out_dir = Path(routes_dir), Path(out_dir)
    points = BufferedParquetWriter(out_dir / "route_points.parquet", ROUTE_POINT_SCHEMA)
    ingested_at = _now()
    for path in sorted(routes_dir.glob("*.gpx")):
        ingest = {"_source_file": path.name, "_ingested_at": ingested_at}
        route_file = f"/workout-routes/{path.name}"  # matches Workout FileReference path
        tree = etree.parse(str(path), etree.XMLParser(huge_tree=True, recover=True))
        for idx, pt in enumerate(tree.iter(f"{_GPX_NS}trkpt")):
            ext = pt.find(f"{_GPX_NS}extensions")

            def x(tag: str, node=ext):
                if node is None:
                    return None
                el = node.find(f"{_GPX_NS}{tag}")
                return el.text if el is not None else None

            points.write(
                {
                    "route_file": route_file,
                    "point_index": idx,
                    "lon": pt.get("lon"),
                    "lat": pt.get("lat"),
                    "ele": x("ele", pt),
                    "time": x("time", pt),
                    "speed": x("speed"),
                    "course": x("course"),
                    "h_acc": x("hAcc"),
                    "v_acc": x("vAcc"),
                    **ingest,
                }
            )
    return {"route_points": points.close()}


def parse_clinical(clinical_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    clinical_dir, out_dir = Path(clinical_dir), Path(out_dir)
    w = BufferedParquetWriter(out_dir / "clinical_records.parquet", CLINICAL_SCHEMA)
    ingested_at = _now()
    for path in sorted(clinical_dir.glob("*.json")):
        raw = path.read_text(encoding="utf-8")
        try:
            resource_type = json.loads(raw).get("resourceType")
        except json.JSONDecodeError:
            resource_type = None
        resource_type = resource_type or path.name.split("-")[0]
        w.write(
            {
                "record_file": path.name,
                "resource_type": resource_type,
                "raw_json": raw,
                "_source_file": path.name,
                "_ingested_at": ingested_at,
            }
        )
    return {"clinical_records": w.close()}
