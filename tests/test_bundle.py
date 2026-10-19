"""Offline checks of databricks.yml (no workspace, no auth)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
BUNDLE = yaml.safe_load((REPO / "databricks.yml").read_text())
JOB = BUNDLE["resources"]["jobs"]["health_pipeline"]


def _cli() -> str | None:
    local = REPO / ".cache" / "databricks-cli" / "databricks"
    return str(local) if local.exists() else shutil.which("databricks")


def test_variables_and_targets():
    assert {"catalog", "schema", "timezone", "warehouse_id"} <= set(BUNDLE["variables"])
    assert BUNDLE["targets"]["dev"]["mode"] == "development"
    assert BUNDLE["targets"]["dev"]["default"] is True
    assert BUNDLE["targets"]["prod"]["mode"] == "production"


def test_no_hardcoded_workspace_host():
    """The host comes from DATABRICKS_HOST / the auth profile, never from the committed bundle."""
    for name, target in BUNDLE["targets"].items():
        assert "host" not in (target.get("workspace") or {}), f"target {name} hard-codes workspace.host"
    assert "host" not in (BUNDLE.get("workspace") or {})


def test_notebook_paths_exist():
    for task in JOB["tasks"]:
        path = REPO / task["notebook_task"]["notebook_path"]
        assert path.exists(), f"{task['task_key']}: {path} missing"
        assert path.read_text().startswith("# Databricks notebook source"), path


def test_task_graph_is_ordered():
    keys = [t["task_key"] for t in JOB["tasks"]]
    assert keys[:5] == ["setup", "bronze", "silver", "gold", "data_quality"]
    seen: set[str] = set()
    for task in JOB["tasks"]:
        for dep in task.get("depends_on", []):
            assert dep["task_key"] in seen, f"{task['task_key']} depends on later task {dep['task_key']}"
        seen.add(task["task_key"])


def test_notebooks_run_common_setup():
    for nb in sorted((REPO / "databricks" / "notebooks").glob("0*.py")):
        assert "# MAGIC %run ./_common" in nb.read_text(), nb


def test_app_and_sync():
    app = BUNDLE["resources"]["apps"]["health_app"]
    assert (REPO / app["source_code_path"]).is_dir()
    assert app["resources"][0]["sql_warehouse"] == {"id": "${var.warehouse_id}", "permission": "CAN_USE"}
    excludes = BUNDLE["sync"]["exclude"]
    for pattern in ("apple_health_export/**", "data/**", ".venv/**", "hf_cache/**", ".cache/**"):
        assert pattern in excludes


def test_bundle_matches_cli_json_schema(tmp_path):
    """`databricks bundle schema` works offline; validate databricks.yml against it."""
    jsonschema = pytest.importorskip("jsonschema")
    cli = _cli()
    if cli is None:
        pytest.skip("databricks CLI not installed")
    env = {**os.environ, "HOME": str(tmp_path)}  # no profile, no auth
    env.pop("DATABRICKS_HOST", None)
    out = subprocess.run([cli, "bundle", "schema"], capture_output=True, text=True, env=env, check=True)
    schema = _python_regex(json.loads(out.stdout))
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(BUNDLE), key=lambda e: str(list(e.path)))
    assert not errors, "\n".join(f"{list(e.path)}: {e.message[:200]}" for e in errors[:10])


def _python_regex(node):
    """The CLI schema uses \\p{L}/\\p{N} (Go regex), which Python's `re` lacks."""
    if isinstance(node, dict):
        return {
            k: (
                v.replace(r"\p{L}", r"[^\W\d_]").replace(r"\p{N}", r"\d")
                if k == "pattern" and isinstance(v, str)
                else _python_regex(v)
            )
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_python_regex(v) for v in node]
    return node
