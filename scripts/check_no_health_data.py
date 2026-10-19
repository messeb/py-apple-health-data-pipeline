#!/usr/bin/env python3
"""Fail if files that would be committed contain Apple Health data or workspace secrets.

A second line of defence behind .gitignore, run by pre-commit and CI:

    python scripts/check_no_health_data.py               # tracked + untracked-but-not-ignored files
    python scripts/check_no_health_data.py --staged      # only what is staged for the next commit
    python scripts/check_no_health_data.py a.csv b.xml   # explicit files (how pre-commit calls it)

It flags
  * export artifacts by name: export.xml, export_cda.xml, export.zip, anything under
    apple_health_export/, electrocardiograms/, workout-routes/ or clinical-records/, *.gpx outside
    tests/fixtures/, *.parquet and *.duckdb files,
  * export artifacts by content: a renamed HealthKit XML export (root element HealthData,
    allowed under tests/fixtures/), an ECG CSV whose
    header has "Date of Birth" next to a name that is not an obvious synthetic one,
  * files larger than --max-mb (default 5 MB),
  * hard-coded Databricks workspace hosts (AWS dbc-..., Azure adb-..., GCP), Databricks
    personal access tokens and Hugging Face tokens. Documentation placeholders such as
    dbc-1234abcd-5678 or adb-1234567890123456 are allowed,
  * any string listed in an optional, git-ignored `.privacy-denylist` file at the repo root
    (one entry per line, e.g. your name, e-mail or device names; matched case-insensitively).

A line containing `privacy-guard: allow` is never reported. Exit code 0 = clean, 1 = findings,
2 = usage error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_MAX_MB = 5.0
CONTENT_SCAN_BYTES = 5 * 1024 * 1024
DENYLIST_FILE = ".privacy-denylist"
ALLOW_MARKER = "privacy-guard: allow"

EXPORT_FILE_NAMES = {"export.xml", "export_cda.xml", "export.zip"}
EXPORT_DIR_NAMES = {"apple_health_export", "electrocardiograms", "workout-routes", "clinical-records"}
DATA_SUFFIXES = {".parquet", ".duckdb"}
GPX_ALLOWED_PREFIX = ("tests", "fixtures")

# Names that obviously belong to synthetic or example data in ECG headers.
SYNTHETIC_NAME = re.compile(
    r"^$|^(jane|john|max|erika)\s+(doe|mustermann|appleseed)$|synthetic|sample|example|demo|test|anonym|redacted|^x+$",
    re.IGNORECASE,
)

# Workspace hosts. Real ids are random; placeholders use runs like 1234 / abcd / xxxx / 0000.
HOST_PATTERNS = [
    re.compile(r"\bdbc-([0-9a-f]{8}-[0-9a-f]{4})\.cloud\.databricks\.com", re.IGNORECASE),
    re.compile(r"\badb-([0-9]{6,})\.[0-9]+\.azuredatabricks\.net", re.IGNORECASE),
    re.compile(r"\b([0-9]{10,})\.[0-9]+\.gcp\.databricks\.com", re.IGNORECASE),
]
PLACEHOLDER_ID = re.compile(r"1234|abcd|x{4}|0000", re.IGNORECASE)
TOKEN_PATTERNS = [
    (re.compile(r"\bdapi[0-9a-f]{32}\b"), "Databricks personal access token"),
    (re.compile(r"\bhf_[A-Za-z0-9]{34,}\b"), "Hugging Face token"),
]
# An XML document (not source code that merely mentions the tag) whose root is HealthData.
HEALTH_XML = re.compile(r"\A\ufeff?\s*(<\?xml[^>]*>\s*)?(<!--.*?-->\s*)*(<!DOCTYPE\s+|<)HealthData\b", re.DOTALL)
XML_FIXTURE_PREFIX = ("tests", "fixtures")
ECG_DOB = re.compile(r"^\"?Date of Birth\"?,", re.MULTILINE)
ECG_NAME = re.compile(r"^\"?Name\"?,\"?([^\"\r\n]*)\"?", re.MULTILINE)


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str
    line: int | None = None

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: {self.reason}"


def check_path_name(rel: str) -> list[Finding]:
    """Checks that only need the repository-relative path."""
    p = PurePosixPath(rel)
    name = p.name.lower()
    found: list[Finding] = []
    if name in EXPORT_FILE_NAMES:
        found.append(Finding(rel, f"Apple Health export file '{p.name}'"))
    dirs = {part.lower() for part in p.parts[:-1]}
    for d in sorted(EXPORT_DIR_NAMES & dirs):
        found.append(Finding(rel, f"file inside an export folder '{d}/'"))
    suffix = p.suffix.lower()
    if suffix in DATA_SUFFIXES:
        found.append(Finding(rel, f"data file '*{suffix}' (pipeline output, never commit)"))
    if suffix == ".gpx" and p.parts[: len(GPX_ALLOWED_PREFIX)] != GPX_ALLOWED_PREFIX:
        found.append(Finding(rel, "GPS route '*.gpx' outside tests/fixtures/"))
    return found


def check_content(rel: str, text: str, denylist: list[str] = ()) -> list[Finding]:
    """Checks on the (decoded) file content."""
    found: list[Finding] = []
    lines = text.splitlines()

    def line_of(pos: int) -> int:
        return text.count("\n", 0, pos) + 1

    def allowed(lineno: int) -> bool:
        return 0 < lineno <= len(lines) and ALLOW_MARKER in lines[lineno - 1]

    is_fixture = PurePosixPath(rel).parts[: len(XML_FIXTURE_PREFIX)] == XML_FIXTURE_PREFIX
    if not is_fixture and HEALTH_XML.search(text[:65536]):
        found.append(Finding(rel, "HealthKit XML export (root element HealthData)", 1))

    if ECG_DOB.search(text):
        for m in ECG_NAME.finditer(text):
            name = m.group(1).strip()
            if not SYNTHETIC_NAME.search(name) and not allowed(line_of(m.start())):
                found.append(
                    Finding(rel, "ECG CSV header with a real-looking name and date of birth", line_of(m.start()))
                )

    for pattern in HOST_PATTERNS:
        for m in pattern.finditer(text):
            n = line_of(m.start())
            if not PLACEHOLDER_ID.search(m.group(1)) and not allowed(n):
                found.append(Finding(rel, f"hard-coded Databricks workspace host '{m.group(0)}'", n))

    for pattern, label in TOKEN_PATTERNS:
        for m in pattern.finditer(text):
            n = line_of(m.start())
            if not allowed(n):
                found.append(Finding(rel, label, n))

    lowered = text.lower()
    for entry in denylist:
        start = lowered.find(entry.lower())
        while start != -1:
            n = line_of(start)
            if not allowed(n):
                found.append(Finding(rel, "matches an entry of .privacy-denylist", n))
            start = lowered.find(entry.lower(), start + 1)
    return found


def check_file(root: Path, rel: str, max_bytes: int, denylist: list[str] = ()) -> list[Finding]:
    found = check_path_name(rel)
    path = root / rel
    if not path.is_file() or path.is_symlink():
        return found
    size = path.stat().st_size
    if size > max_bytes:
        found.append(Finding(rel, f"file is {size / 1e6:.1f} MB (limit {max_bytes / 1e6:.1f} MB)"))
    with path.open("rb") as fh:
        raw = fh.read(CONTENT_SCAN_BYTES)
    if b"\0" in raw[:8192]:
        return found  # binary
    found += check_content(rel, raw.decode("utf-8", errors="replace"), denylist)
    return found


def _git(root: Path, *args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.split("\0") if line]


def candidate_files(root: Path, staged: bool) -> list[str]:
    if staged:
        return _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return sorted(set(_git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")))


def load_denylist(root: Path) -> list[str]:
    path = root / DENYLIST_FILE
    if not path.is_file():
        return []
    entries = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [e for e in entries if e and not e.startswith("#")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", help="files to check (default: every file git would commit)")
    parser.add_argument("--staged", action="store_true", help="check only the files staged for commit")
    parser.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB, help="largest allowed file size")
    parser.add_argument("--root", type=Path, default=None, help="repository root (default: git top level)")
    args = parser.parse_args(argv)

    root = args.root
    if root is None:
        try:
            root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel")[0].strip())
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
            root = Path.cwd()
    root = root.resolve()

    if args.files:
        files = []
        for f in args.files:
            p = Path(f).resolve()
            files.append(p.relative_to(root).as_posix() if p.is_relative_to(root) else f)
    else:
        try:
            files = candidate_files(root, args.staged)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{root} is not a git repository; pass the files to check explicitly.", file=sys.stderr)
            return 2

    denylist = load_denylist(root)
    findings = [
        f for rel in files if rel != DENYLIST_FILE for f in check_file(root, rel, int(args.max_mb * 1e6), denylist)
    ]
    for finding in findings:
        print(finding, file=sys.stderr)
    if findings:
        print(
            f"\n{len(findings)} privacy finding(s). Personal health data and workspace details must never be "
            "committed.\nUse the synthetic generator (`health-lakehouse sample`) for examples, or add "
            f"'{ALLOW_MARKER}' to a line that is a false positive.",
            file=sys.stderr,
        )
        return 1
    print(f"privacy guard: {len(files)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
