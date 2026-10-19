# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup: schema, volume, raw upload
# MAGIC
# MAGIC Creates the Unity Catalog schema and the managed volume for the raw export, then checks
# MAGIC that the Apple Health export is there. If only a zip was uploaded, it gets unpacked.
# MAGIC
# MAGIC **Upload the export (one of):**
# MAGIC
# MAGIC 1. CLI, folder (recommended, after this notebook has created the volume):
# MAGIC    `databricks fs cp -r ./apple_health_export dbfs:/Volumes/<catalog>/<schema>/raw/apple_health_export`
# MAGIC 2. CLI, zip: `databricks fs cp ./export.zip dbfs:/Volumes/<catalog>/<schema>/raw/export.zip`
# MAGIC 3. UI: *Catalog → <catalog> → <schema> → Volumes → raw → Upload to this volume*
# MAGIC    (upload `export.zip`; the UI does not upload folders).
# MAGIC
# MAGIC The expected layout is `/Volumes/<catalog>/<schema>/raw/apple_health_export/export.xml`
# MAGIC plus the optional `electrocardiograms/`, `workout-routes/` and `clinical-records/` folders.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {FQ} COMMENT 'Personal health lakehouse from an Apple Health export (bronze/silver/gold)'"
)
spark.sql(
    f"CREATE VOLUME IF NOT EXISTS {FQ}.`{VOLUME}` "
    "COMMENT 'Raw Apple Health export (export.xml, ECG CSVs, GPX routes, FHIR JSON)'"
)
print(f"ready: schema {CATALOG}.{SCHEMA}, volume {VOLUME_ROOT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unpack `export.zip` if needed
# MAGIC
# MAGIC Looks for `export.zip` (or any single `*.zip`) in the volume root. The iPhone's zip contains
# MAGIC an `apple_health_export/` folder; members are written into `EXPORT_DIR` either way.
# MAGIC Skipped when `export.xml` is already present, so the notebook is idempotent.

# COMMAND ----------

import os
import shutil
import zipfile


def find_zip() -> str | None:
    zips = sorted(f for f in os.listdir(VOLUME_ROOT) if f.lower().endswith(".zip"))
    if "export.zip" in zips:
        return f"{VOLUME_ROOT}/export.zip"
    return f"{VOLUME_ROOT}/{zips[0]}" if len(zips) == 1 else None


def unzip_export(zip_path: str, target: str) -> int:
    """Stream members to the volume (zip members are read sequentially, which FUSE handles well)."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or "__MACOSX" in info.filename:
                continue
            rel = info.filename.split("apple_health_export/", 1)[-1]
            dest = os.path.join(target, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            n += 1
    return n


if os.path.exists(f"{EXPORT_DIR}/export.xml"):
    print(f"export.xml already in {EXPORT_DIR}; nothing to unpack")
elif (zip_path := find_zip()) is not None:
    print(f"unpacking {zip_path} -> {EXPORT_DIR} (a 3 GB export takes a few minutes)")
    print(f"{unzip_export(zip_path, EXPORT_DIR)} files written")
else:
    print(f"no export found yet: upload it to {EXPORT_DIR}/ or {VOLUME_ROOT}/export.zip (see above)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check the upload

# COMMAND ----------

xml = f"{EXPORT_DIR}/export.xml"
if not os.path.exists(xml):
    raise FileNotFoundError(f"{xml} is missing - upload the Apple Health export first (see the top of this notebook)")
print(f"export.xml: {os.path.getsize(xml) / 1e9:.2f} GB")
for sub in ("electrocardiograms", "workout-routes", "clinical-records"):
    path = f"{EXPORT_DIR}/{sub}"
    print(f"{sub:<20} {len(os.listdir(path)) if os.path.isdir(path) else 0:>6} files")
