"""Shared fixtures: a synthetic Apple Health export run through the whole pipeline.

Tests never read the real `data/` or `apple_health_export/`. The session builds a
small synthetic lakehouse once (export -> bronze -> silver -> gold, plus the ML
`--quick` outputs) in a temporary directory and the app, ML and Spark tests
point at it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SYNTH_DAYS = 400
SYNTH_SEED = 7


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """`<root>/apple_health_export` and `<root>/data/{bronze,silver,gold}` from the generator."""
    from health_lakehouse.cli import run_pipeline
    from health_lakehouse.sample import generate

    root = tmp_path_factory.mktemp("synthetic")
    generate(root / "apple_health_export", days=SYNTH_DAYS, seed=SYNTH_SEED)
    run_pipeline("all", root / "apple_health_export", root / "data", timezone="auto")
    return root


@pytest.fixture(scope="session")
def synthetic_export(synthetic_root) -> Path:
    return synthetic_root / "apple_health_export"


@pytest.fixture(scope="session")
def synthetic_bronze(synthetic_root) -> Path:
    return synthetic_root / "data" / "bronze"


@pytest.fixture(scope="session")
def synthetic_data_dir(synthetic_root) -> Path:
    """Lakehouse root (silver/gold/ml) with the ML `--quick` outputs, as HEALTH_DATA_DIR expects."""
    pytest.importorskip("torch")
    data = synthetic_root / "data"
    if not (data / "ml").is_dir():
        env = {
            **os.environ,
            "HEALTH_DATA_DIR": str(data),
            "HEALTHML_REPORT_DIR": str(synthetic_root / "reports"),
            "HEALTHML_DEVICE": os.environ.get("HEALTHML_DEVICE", "cpu"),
            "MPLBACKEND": "Agg",
        }
        subprocess.run(
            [sys.executable, str(REPO / "ml" / "run_all.py"), "--quick"],
            env=env,
            check=True,
            cwd=REPO,
            stdout=subprocess.DEVNULL,
        )
    return data
