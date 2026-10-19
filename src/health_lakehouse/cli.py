"""Command line entry point.

health-lakehouse {bronze,silver,gold,all} [--export-dir DIR] [--data-dir DIR] [--timezone auto|ZONE]
health-lakehouse sample [--out DIR] [--days N] [--seed N] [--tz ZONE]
health-lakehouse demo   [--dir sample_data]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from .bronze import build_bronze
from .pipeline import build_layers
from .timezone import resolve_timezone


def _pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--export-dir", default="apple_health_export", type=Path)
    p.add_argument("--data-dir", default="data", type=Path)
    p.add_argument(
        "--timezone",
        default="auto",
        help="IANA timezone for local dates, or 'auto' (most common HKTimeZone in the export)",
    )
    p.add_argument("--skip-xml", action="store_true", help="bronze: only parse ECG/routes/clinical")


def run_pipeline(step: str, export_dir: Path, data_dir: Path, timezone: str = "auto", skip_xml: bool = False) -> None:
    bronze_dir = data_dir / "bronze"
    db_path = data_dir / "health.duckdb"
    if step in ("bronze", "all"):
        print("== bronze ==")
        print(build_bronze(export_dir, bronze_dir, skip_xml=skip_xml))
    layers = {"silver": ("silver",), "gold": ("gold",), "all": ("silver", "gold")}.get(step)
    if layers:
        tz = resolve_timezone(timezone, bronze_dir)
        print(f"== {' + '.join(layers)} ==")
        build_layers(db_path, bronze_dir, layers, {"TIMEZONE": tz}, export_parquet_dir=data_dir)


def _sample(args) -> None:
    from .sample import generate

    counts = generate(args.out, days=args.days, seed=args.seed, tz=args.tz, end=args.end)
    print(
        f"{sum(v for k, v in counts.items() if k.startswith('HK')):,} records, {counts.get('Workout', 0)} workouts, "
        f"{counts['routes']} routes, {counts['ecgs']} ECGs, {counts['clinical_records']} clinical records"
    )


def _demo(args) -> None:
    from .sample import generate

    root = args.dir
    export_dir, data_dir = root / "apple_health_export", root / "data"
    print(f"== sample -> {export_dir} ==")
    generate(export_dir, days=args.days, seed=args.seed)
    run_pipeline("all", export_dir, data_dir)
    run_all = Path("ml") / "run_all.py"
    if args.skip_ml:
        pass
    elif run_all.exists():
        print("== ml (quick) ==")
        env = {
            **os.environ,
            "HEALTH_DATA_DIR": str(data_dir.resolve()),
            "HEALTHML_REPORT_DIR": str((root / "reports").resolve()),
        }
        subprocess.run([sys.executable, str(run_all), "--quick"], env=env, check=True)
    else:
        print("(ml/run_all.py not found in the current directory: skipping the ML step)")
    print(
        f"\nDone. Start the Streamlit app on the synthetic data with:\n\n"
        f"    HEALTH_DATA_DIR={data_dir} streamlit run app/app.py\n"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="health-pipeline", description="Apple Health export -> bronze/silver/gold lakehouse"
    )
    sub = p.add_subparsers(dest="step", required=True)
    for step in ("bronze", "silver", "gold", "all"):
        _pipeline_args(
            sub.add_parser(step, help=f"build the {step} layer" if step != "all" else "bronze + silver + gold")
        )

    s = sub.add_parser("sample", help="write a synthetic Apple Health export (fictional person)")
    s.add_argument("--out", default=Path("sample_data/apple_health_export"), type=Path)
    s.add_argument("--days", default=730, type=int)
    s.add_argument("--seed", default=42, type=int)
    s.add_argument("--tz", default="Europe/Berlin", help="IANA zone the synthetic person lives in")
    s.add_argument("--end", default=None, type=date.fromisoformat, help="last day (YYYY-MM-DD); fixed default")

    d = sub.add_parser("demo", help="sample export -> pipeline -> ML --quick, all under --dir")
    d.add_argument("--dir", default=Path("sample_data"), type=Path)
    d.add_argument("--days", default=730, type=int)
    d.add_argument("--seed", default=42, type=int)
    d.add_argument("--skip-ml", action="store_true")

    args = p.parse_args(argv)
    if args.step == "sample":
        if args.end is None:
            from .sample import DEFAULT_END

            args.end = DEFAULT_END
        _sample(args)
    elif args.step == "demo":
        _demo(args)
    else:
        run_pipeline(args.step, args.export_dir, args.data_dir, args.timezone, args.skip_xml)


if __name__ == "__main__":
    main()
