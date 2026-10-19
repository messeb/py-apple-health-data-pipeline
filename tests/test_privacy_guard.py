"""scripts/check_no_health_data.py: the pre-commit / CI guard against committing health data."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_no_health_data", REPO / "scripts" / "check_no_health_data.py")
guard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = guard  # dataclasses look the module up
_spec.loader.exec_module(guard)

# Built from parts so this file never contains a literal host or token the guard would flag.
REAL_AWS_HOST = "https://dbc-" + "7f3e9c21-a6b4" + ".cloud.databricks.com"
REAL_AZURE_HOST = "https://adb-" + "8350192837465" + ".2.azuredatabricks.net"
PAT = "dapi" + "5f" * 16


def reasons(findings):
    return [f.reason for f in findings]


@pytest.mark.parametrize(
    "rel",
    [
        "export.xml",
        "apple_health_export/export_cda.xml",
        "uploads/export.zip",
        "apple_health_export/electrocardiograms/ecg_2024-01-15.csv",
        "foo/workout-routes/route_1.gpx",
        "data/silver/records.parquet",
        "data/health.duckdb",
        "notes/ride.gpx",
    ],
)
def test_export_artifacts_are_flagged_by_name(rel):
    assert guard.check_path_name(rel), rel


@pytest.mark.parametrize(
    "rel", ["src/health_lakehouse/bronze/xml_records.py", "tests/fixtures/routes/route_1.gpx", "docs/export.md"]
)
def test_ordinary_paths_pass(rel):
    assert guard.check_path_name(rel) == []


def test_renamed_healthkit_xml_is_flagged():
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE HealthData [\n<!ELEMENT HealthData ANY>\n]>\n'
    assert guard.check_content("backup/my_health.xml", xml + "<HealthData locale='en_US'/>")
    assert guard.check_content("backup/my_health.xml", "<HealthData locale='en_US'>\n</HealthData>")


def test_source_code_mentioning_healthdata_and_fixtures_pass():
    assert guard.check_content("src/parser.py", 'ROOT = "<HealthData>"\n') == []
    assert guard.check_content("tests/fixtures/export.xml", "<?xml version='1.0'?>\n<HealthData/>") == []


def _ecg(name: str) -> str:
    return "\n".join([f"Name,{name}", 'Date of Birth,"1. Jan 1990"', "Classification,Sinus Rhythm", "Unit,µV", "1,2"])


def test_ecg_with_real_looking_name_is_flagged():
    assert "ECG CSV header" in reasons(guard.check_content("ecg.csv", _ecg("Pat Smith")))[0]


@pytest.mark.parametrize(
    "name", ["Jane Doe", "Jane Appleseed", "Max Mustermann", "Synthetic Person", "Sample User", ""]
)
def test_ecg_with_synthetic_name_passes(name):
    assert guard.check_content("ecg.csv", _ecg(name)) == []


def test_workspace_hosts_and_tokens_are_flagged():
    text = f"host: {REAL_AWS_HOST}\nazure: {REAL_AZURE_HOST}\ntoken = {PAT}\n"
    found = guard.check_content("databricks.yml", text)
    assert [f.line for f in found] == [1, 2, 3]
    assert "workspace host" in found[0].reason and "access token" in found[2].reason


def test_doc_placeholders_and_allow_marker_pass():
    docs = (
        "- `https://dbc-1234abcd-5678.cloud.databricks.com` (AWS)\n"
        "- `https://adb-1234567890123456.7.azuredatabricks.net` (Azure)\n"
        "- `https://<your-workspace>.cloud.databricks.com`\n"
        f"- {REAL_AWS_HOST}  # privacy-guard: allow\n"
    )
    assert guard.check_content("docs/databricks.md", docs) == []


def test_denylist_entries_are_flagged_case_insensitively():
    found = guard.check_content("notes.md", "line one\nRecorded on Kim's Apple Watch\n", ["kim's apple watch"])
    assert [(f.line, f.reason) for f in found] == [(2, "matches an entry of .privacy-denylist")]


def test_large_and_binary_files(tmp_path):
    (tmp_path / "big.bin").write_bytes(b"\0" * 2048)
    (tmp_path / "img.png").write_bytes(b"\x89PNG\0\0" + REAL_AWS_HOST.encode())
    assert reasons(guard.check_file(tmp_path, "big.bin", max_bytes=1024)) == ["file is 0.0 MB (limit 0.0 MB)"]
    assert guard.check_file(tmp_path, "img.png", max_bytes=10_000) == []  # binary content is not scanned


def test_main_on_explicit_files(tmp_path, capsys):
    (tmp_path / "ok.py").write_text("print('hello')\n")
    (tmp_path / "bad.yml").write_text(f"host: {REAL_AWS_HOST}\n")
    assert guard.main(["--root", str(tmp_path), str(tmp_path / "ok.py")]) == 0
    assert guard.main(["--root", str(tmp_path), str(tmp_path / "ok.py"), str(tmp_path / "bad.yml")]) == 1
    assert "bad.yml:1: hard-coded Databricks workspace host" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_main_respects_gitignore(tmp_path):
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("data/\nsample_data/\n")
    for rel in ("data/health.duckdb", "sample_data/data/silver/records.parquet"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"\0")
    (tmp_path / "README.md").write_text("hello\n")
    assert guard.main(["--root", str(tmp_path)]) == 0  # ignored synthetic/real outputs are never candidates

    (tmp_path / "leak.parquet").write_bytes(b"\0")
    assert guard.main(["--root", str(tmp_path)]) == 1
    subprocess.run([*git, "add", "README.md"], cwd=tmp_path, check=True)
    assert guard.main(["--root", str(tmp_path), "--staged"]) == 0  # leak.parquet is not staged


def test_repository_itself_is_clean():
    """Every file in this checkout that git would commit passes the guard."""
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    assert guard.main(["--root", str(REPO)]) == 0
