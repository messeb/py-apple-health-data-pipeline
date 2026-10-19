"""Bronze layer: raw, untyped copies of every file in the Apple Health export."""

from __future__ import annotations

from pathlib import Path

from .export_xml import parse_export_xml
from .side_files import parse_clinical, parse_ecgs, parse_routes

__all__ = ["build_bronze", "parse_clinical", "parse_ecgs", "parse_export_xml", "parse_routes"]


def build_bronze(export_dir: str | Path, bronze_dir: str | Path, skip_xml: bool = False) -> dict[str, int]:
    """Parse a whole `apple_health_export/` folder into `bronze_dir/*.parquet`."""
    export_dir, bronze_dir = Path(export_dir), Path(bronze_dir)
    counts: dict[str, int] = {}
    if (export_dir / "electrocardiograms").is_dir():
        counts |= parse_ecgs(export_dir / "electrocardiograms", bronze_dir)
    if (export_dir / "clinical-records").is_dir():
        counts |= parse_clinical(export_dir / "clinical-records", bronze_dir)
    if (export_dir / "workout-routes").is_dir():
        counts |= parse_routes(export_dir / "workout-routes", bronze_dir)
    if not skip_xml:
        counts |= parse_export_xml(export_dir / "export.xml", bronze_dir)
    return counts
