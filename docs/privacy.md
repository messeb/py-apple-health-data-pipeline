# Privacy

Your Apple Health export is some of the most personal data you have. This project is built so that it never
leaves your control.

## What the Pipeline Does

- **Everything runs where you run it.** Locally, data stays in the project folder. On Databricks, it stays
  in your own workspace and Unity Catalog schema. It has no server, telemetry or analytics.
- **One optional download.** The ML suite can download one public model from Hugging Face
  (`amazon/chronos-bolt-small`, ≈190 MB). That is the only outgoing request, and it sends none of your data.
  `--quick` skips it.
- **Personal identifiers are dropped early.** ECG files contain your full name and date of birth, and the
  bronze parser drops both before anything is written.
- **Git can't pick up your data by accident:**
  - `.gitignore` excludes `apple_health_export/`, `export.zip`, `data/`, `sample_data/`, Parquet/DuckDB
    files, ML reports and caches.
  - A pre-commit hook (`scripts/check_no_health_data.py`) blocks export files, Parquet/DuckDB files, large
    files and hard-coded Databricks workspace URLs.
- **The Databricks bundle uploads code only.** `databricks bundle deploy` excludes the export and all local
  data. You upload the export yourself, into a volume in your own schema.

## What Is in an Export

Know what is in your export before you share anything derived from it:

| Where | What | What the pipeline does |
| --- | --- | --- |
| `electrocardiograms/*.csv` header | Your **full name** and **date of birth** | Dropped in bronze |
| `export.xml` `<Me>` element | Date of birth, biological sex, blood type, skin type | Kept in `bronze_me` / `silver_profile` (some analyses need age or sex); drop the tables if you don't want them |
| `sourceName` / `device` attributes | Device names, often with your first name ("Kim's Apple Watch") | Kept in `source_name` |
| `workout-routes/*.gpx` | Precise **GPS tracks**, which reveal home, work and routines | Kept in `silver_route_points` |
| `clinical-records/*.json` | FHIR medical records: diagnoses, medications, lab results, providers | Kept in bronze |
| `export_cda.xml` | The same health data in HL7 CDA format | Not read |
| `export.xml` `<Record>`s | Every heart rate, sleep, cycle tracking, mindfulness and medication sample | The basis of every table |

App and dashboard screenshots, ML figures, `ml/reports/` and Parquet/DuckDB files contain the same
information in aggregated form. Treat them as personal data too.

To make the pre-commit guard also catch your name, e-mail or device names, list them in a git-ignored
`.privacy-denylist` file at the repository root.

## Recommendations

- **Databricks.** Keep the schema private: grant only yourself and the app's service principal.
  Don't share the app URL publicly; Databricks Apps require workspace login by default.
- **Screenshots and bug reports.** Use the synthetic data (`make demo`). Never attach a real export,
  Parquet files or app screenshots of real data to an issue.
- **Delete.** To delete everything:
  - locally: remove `apple_health_export/` and `data/`;
  - on Databricks: run `DROP SCHEMA <catalog>.<schema> CASCADE`.

Found a privacy problem? See [SECURITY.md](../SECURITY.md).
