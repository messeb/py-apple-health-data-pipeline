# Step by Step: Apple Health Data Pipeline on Databricks

This guide goes from an empty workspace to running tables, ML results, the Lakeview dashboard and the
Streamlit app (as a Databricks App). It was built and tested on the free
[Databricks Free Edition](https://www.databricks.com/learn/free-edition), so you don't need a paid
workspace; the defaults below match it.

Do the steps in order. Each step ends with a **Check** so you know it worked before you go on.

Run all commands from the root of your clone of this repository:

```bash
git clone https://github.com/messeb/py-apple-health-data-pipeline.git
cd py-apple-health-data-pipeline
```

The guide uses these names. If you use other ones, replace them everywhere.

| Thing | Value used here |
|---|---|
| Catalog | `workspace` (Free Edition and new workspaces; older ones often use `main`) |
| Schema | `health` |
| Volume for the raw export | `raw` |
| Timezone | `auto` (detected from the export's `HKTimeZone` metadata; set e.g. `Europe/Berlin` to override) |
| Bundle target | `dev` |

Time needed: about 30 minutes, most of it the upload (a multi-year export is several GB).

---

## Step 1: Check the Workspace

In the Databricks web UI, check four things:

1. **Unity Catalog** is enabled: the left sidebar has **Catalog**. The project uses the catalog
   `workspace`; `databricks catalogs list` shows yours. If yours has another name (older workspaces
   often use `main`), set it as `default:` of the `catalog` variable in `databricks.yml` and as
   `HEALTH_CATALOG` in `app/app.yaml`, and replace `workspace` in the commands of this guide.
2. **Serverless compute** for jobs is available: *Settings → Workspace admin → Previews/Compute*. On
   most current workspaces it is on by default.
3. **Databricks Apps** is available: the **+ New** menu has **App**.
4. A **SQL warehouse** exists: *SQL Warehouses* in the sidebar. New workspaces have one called
   **Serverless Starter Warehouse**. The bundle looks for exactly that name. If yours has a different
   name, copy its **ID** from the warehouse's *Overview* tab; you'll need it in step 5.

**Check:** you can open *Catalog → workspace* and see a SQL warehouse in the list.

---

## Step 2: Install the Databricks CLI

```bash
brew tap databricks/tap
brew install databricks
databricks --version
```

**Check:** the version is v0.218 or newer.

---

## Step 3: Log In

Copy your workspace URL from the browser address bar. It looks like one of these:

- `https://dbc-1234abcd-5678.cloud.databricks.com` (AWS)
- `https://adb-1234567890123456.7.azuredatabricks.net` (Azure)
- `https://1234567890123456.7.gcp.databricks.com` (GCP)

Then log in:

```bash
export DATABRICKS_HOST=https://<your-workspace-url>      # paste your URL, no trailing slash
databricks auth login --host $DATABRICKS_HOST
```

A browser window opens. Log in there, then go back to the terminal. When the CLI asks for a profile
name, accept the suggested one.

**Check:**

```bash
databricks current-user me
```

This prints your user name.

---

## Step 4: Point the Bundle at Your Workspace

`databricks.yml` contains no workspace URL. The CLI uses the workspace you logged in to in step 3, so
there is nothing to edit. If you have several CLI profiles, choose one for the whole session:

```bash
export DATABRICKS_CONFIG_PROFILE=<profile-name>     # names: databricks auth profiles
```

**Check:** `databricks bundle validate -t dev` prints your workspace URL under `Workspace: Host:`.

---

## Step 5: Validate and Deploy the Bundle

```bash
databricks bundle validate -t dev
databricks bundle deploy   -t dev
```

If your warehouse is not called "Serverless Starter Warehouse", or you use another catalog, add the
variables to **both** commands:

```bash
databricks bundle deploy -t dev --var warehouse_id=<warehouse-id> --var catalog=<catalog>
```

What `deploy` does:

- It uploads the code (`src/`, `databricks/`, `ml/`, `app/`, …) to
  `/Workspace/Users/<you>/.bundle/health_data_pipeline/dev/files`.
- It creates the job, the Lakeview dashboard and the app.
- It does **not** upload health data: `apple_health_export/` and `data/` are excluded.

**Check:**

- `validate` ends with `Validation OK!`.
- In the UI, *Jobs & Pipelines* shows **[dev &lt;you&gt;] health_pipeline**.
- *Compute → Apps* lists `health-app` (stopped for now).
- `databricks bundle summary -t dev` lists all three and gives their URLs and exact names.

---

## Step 6: Create the Schema and the Volume

```bash
databricks schemas create health workspace
databricks volumes create workspace health raw MANAGED
```

If the schema already exists, you get an "already exists" error. That's fine; go on.

**Check:** *Catalog → workspace → health → Volumes* shows `raw`.

---

## Step 7: Upload the Apple Health Export

Pick **one** of the options below.

### Option A: The Unpacked Folder You Already Have (CLI)

`export_cda.xml` (1.4 GB) is not used by the pipeline, so skip it. That saves a third of the upload.

```bash
V=dbfs:/Volumes/workspace/health/raw/apple_health_export

databricks fs mkdir $V
databricks fs cp apple_health_export/export.xml $V/export.xml
databricks fs cp -r apple_health_export/electrocardiograms $V/electrocardiograms
databricks fs cp -r apple_health_export/workout-routes     $V/workout-routes
databricks fs cp -r apple_health_export/clinical-records   $V/clinical-records
```

### Option B: The Zip from the iPhone (CLI or UI)

On the iPhone, go to *Health → your profile picture → Export All Health Data*. This gives you
`export.zip`. Then either upload it with the CLI:

```bash
databricks fs cp export.zip dbfs:/Volumes/workspace/health/raw/export.zip
```

or use the UI: *Catalog → workspace → health → Volumes → raw → Upload to this volume*. The UI accepts
files only, not folders, so use the zip there. The first pipeline task unzips it for you.

**Check:**

```bash
databricks fs ls dbfs:/Volumes/workspace/health/raw/apple_health_export
```

You should see `export.xml` and the three folders. With option B, you only see `export.zip` in
`.../raw/`; it is unpacked in step 8.

---

## Step 8: Run the Pipeline

```bash
databricks bundle run -t dev health_pipeline
```

The command prints a link to the run and waits until it finishes, usually in about 10–20 minutes. The
ML tasks take the longest. You can follow the run in the UI through the link.

| # | Task | What happens |
|---|---|---|
| 1 | `setup` | Creates the schema and volume if they are missing, unzips `export.zip` if needed, and checks that `export.xml` exists |
| 2 | `bronze` | Parses the export, writes 12 `bronze_*` tables |
| 3 | `silver` | 12 typed `silver_*` tables (correct local time, units, duplicates removed) |
| 4 | `gold` | 10 `gold_*` tables (daily/weekly summaries, workouts, heart rate, ECG) |
| 5 | `data_quality` | 37 checks, written to `data_quality_results`. The job stops here if a check fails |
| 6 | 5 ML tasks, in parallel | Forecasts, unusual days, day/workout types, ECG beats, classifiers. They write `ml_*` tables and log to MLflow under `/Shared/health_ml` |

Useful variations:

```bash
# other timezone (e.g. after moving):
databricks bundle run -t dev health_pipeline --params timezone=America/New_York
# only rebuild the SQL layers, without ML:
databricks bundle run -t dev health_pipeline --only silver,gold,data_quality
```

**Check:** in a SQL editor (*SQL Editor* in the sidebar, using your warehouse), run:

```sql
SELECT COUNT(*) FROM workspace.health.silver_records;            -- millions for a multi-year export
SELECT MIN(local_date), MAX(local_date), COUNT(*)
FROM workspace.health.gold_daily_summary;                         -- first to last recorded day
SELECT check_name, severity, ok, value FROM workspace.health.data_quality_results
WHERE run_at = (SELECT MAX(run_at) FROM workspace.health.data_quality_results);  -- ok = true everywhere
SHOW TABLES IN workspace.health LIKE 'ml_*';
```

If a task fails, open the run, click the red task and read the error at the bottom of the notebook
output. The most common causes:

- The export is not at `/Volumes/workspace/health/raw/apple_health_export/export.xml` (see step 7).
- Missing rights to create tables in the schema.

Fix the cause, then use **Repair run** in the UI. It reruns only the failed tasks.

---

## Step 9: Open the Lakeview Dashboard

The bundle also deploys a native Databricks **AI/BI (Lakeview) dashboard**,
`databricks/dashboards/health.lvdash.json`. It has five pages (Overview, Activity, Heart & Fitness,
Workouts, Body & Gait) and a global date filter. It needs no app and no grants, because it runs as you
on the SQL warehouse.

```bash
databricks bundle summary -t dev      # shows the dashboard URL under "Dashboards"
```

Or open *Dashboards* in the sidebar and pick **[dev &lt;you&gt;] Health Overview**.

Its queries use plain table names such as `gold_daily_summary`. The bundle points them at your
`catalog`/`schema` (`dataset_catalog` / `dataset_schema` in `databricks.yml`), so the same file works
in every workspace.

**Change it:** edit the dashboard in the UI, then copy the changes back into the repo so the next
deploy doesn't overwrite them:

```bash
databricks bundle generate dashboard --resource health_overview --force -t dev
```

**Import it without the bundle:** *Dashboards → Create → Import dashboard from file*, choose
`health.lvdash.json`, then set catalog and schema on each dataset (*Data* tab) to `workspace` / `health`.

**Check:** the Overview page shows step counts, resting heart rate and HRV for your data.

---

## Step 10: Start the Streamlit App

```bash
databricks bundle run -t dev health_app
```

This uploads the `./app` code into the Databricks App and starts it. It takes a few minutes the first time.

Find the app's exact name. In dev mode it may have a prefix, so look for "health-app":

```bash
databricks apps list
databricks apps get <app-name>
```

Note two values from the output: **`url`** and **`service_principal_client_id`**.

**Check:** the app shows as **RUNNING** under *Compute → Apps*.

---

## Step 11: Give the App Read Access to Your Tables

The app reads the tables as its own service principal, not as you. Until you grant access, the
app shows permission errors. Run this in the SQL editor, with the ID from step 10:

```sql
GRANT USE CATALOG ON CATALOG workspace TO `<service_principal_client_id>`;
GRANT USE SCHEMA, SELECT ON SCHEMA workspace.health TO `<service_principal_client_id>`;
```

The back-ticks are required. `SELECT` on the schema also covers tables created later, such as new
`ml_*` tables.

The app's access to the SQL warehouse (`CAN_USE`) is already set by the bundle.

**Check:** open the app `url` from step 10 and reload. All 8 pages show data, including the ML insights
page. If the page still shows a permission error, wait a minute and reload.

---

## Step 12: Keep Costs Low

- **The app** bills while it runs. Stop it when you don't need it and start it again later:

  ```bash
  databricks apps stop  <app-name>
  databricks apps start <app-name>
  ```

- **SQL warehouse:** *SQL Warehouses → your warehouse → Edit*, set **Auto stop** to 5–10 minutes.
- **The job** runs only when you start it. Serverless compute bills only while the tasks run.

---

## Updating with a New Export Later

1. Export again on the iPhone and upload it as in step 7, replacing the old files.
   With the zip, also delete the old `apple_health_export/` folder in the volume. Otherwise `setup`
   sees the old `export.xml` and doesn't unpack the new zip:

   ```bash
   databricks fs rm -r dbfs:/Volumes/workspace/health/raw/apple_health_export
   ```

2. `databricks bundle run -t dev health_pipeline`. Every task overwrites its tables, so
   nothing is duplicated.
3. The app and the Lakeview dashboard show the new data immediately; no restart is needed.

For automatic nightly runs, remove the `#` in front of the `schedule:` block in `databricks.yml` and
deploy with the `prod` target (dev mode pauses schedules):

```bash
databricks bundle deploy -t prod
```

If you change code (SQL, notebooks, app), run `databricks bundle deploy -t dev` again. For
app changes, also run `databricks bundle run -t dev health_app`.

---

## Removing Everything

```bash
databricks bundle destroy -t dev        # deletes the job, the dashboard, the app and the uploaded code
```

The tables and the uploaded export stay. To delete them as well (this cannot be undone):

```sql
DROP SCHEMA workspace.health CASCADE;
```

---

## Troubleshooting

See the [Databricks section of Troubleshooting](troubleshooting.md#databricks).
