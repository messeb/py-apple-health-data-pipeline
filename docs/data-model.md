# Data Model

All tables exist under the same names locally and on Databricks:

- locally: `data/<layer>/<table>.parquet` and inside `data/health.duckdb`;
- on Databricks: `<catalog>.<schema>.<table>`.

Dates and hours are **local time** in your timezone (see [Architecture](architecture.md#correct-local-time)).

## Bronze: Raw

Bronze keeps a raw copy of each source. Every column is a string, and every table has two lineage columns,
`_source_file` and `_ingested_at`.

| Table | Source | Notes |
|---|---|---|
| `bronze_records` | every `<Record>` in export.xml | quantity and category samples; `metadata_json`; `correlation_id` links blood-pressure pairs; `record_id` is a stable hash |
| `bronze_hrv_instantaneous_bpm` | `<InstantaneousBeatsPerMinute>` inside HRV records | beat-to-beat BPM |
| `bronze_correlations` | `<Correlation>` | e.g. blood pressure |
| `bronze_workouts` | `<Workout>` | `workout_id` is a stable hash; `route_file` from `<WorkoutRoute>` |
| `bronze_workout_events` | `<WorkoutEvent>` | pause, resume, lap, segment |
| `bronze_workout_statistics` | `<WorkoutStatistics>` | energy, distance, heart rate per workout |
| `bronze_activity_summaries` | `<ActivitySummary>` | daily rings and goals |
| `bronze_me` | `<Me>` | date of birth, biological sex, blood type, … |
| `bronze_ecg_readings`, `bronze_ecg_samples` | `electrocardiograms/*.csv` | header and voltage samples. **Name and date of birth are dropped here** |
| `bronze_route_points` | `workout-routes/*.gpx` | GPS points |
| `bronze_clinical_records` | `clinical-records/*.json` | raw FHIR JSON |

`export_cda.xml` duplicates `export.xml` in HL7 CDA format and is not ingested.

## Silver: Typed and Cleaned

| Table | Grain | Key columns |
|---|---|---|
| `silver_records` | one sample | `record_id`, `kind` (quantity/category), `metric` (e.g. `StepCount`, `HeartRate`), `value_num`, `value_str`, `unit`, `source_name`, `device_class` (watch/phone/other), `start_utc`, `start_local`, `end_local`, `local_date`, `local_hour`, `duration_s`, `metadata_json` |
| `silver_sleep_segments` | one sleep interval | `stage` (in_bed/core/deep/rem/awake/asleep_unspecified), `start_local`, `end_local`, `duration_min`, `night_date` |
| `silver_workouts` | one workout | `activity`, `start_local`, `duration_min`, `distance_km`, `active_kcal`, `hr_avg/min/max`, `is_indoor`, `avg_mets`, `weather_temp_c`, `weather_humidity_pct`, `elevation_ascended_m`, `route_file` |
| `silver_workout_statistics`, `silver_workout_events` | per workout | typed statistics and events |
| `silver_activity_summaries` | day | move/exercise/stand values, goals, `*_ring_closed` |
| `silver_hrv_beats` | beat | `sdnn_ms`, `bpm` per HRV window |
| `silver_ecg_readings`, `silver_ecg_samples` | ECG / sample | `classification`, `sample_rate_hz`; `value_uv` in µV |
| `silver_route_points` | GPS point | `lat`, `lon`, `elevation_m`, `point_local`, `speed_mps`, accuracy |
| `silver_clinical_records` | FHIR resource | e.g. vaccine, date, dose |
| `silver_profile` | attribute | characteristics as key/value |

Silver also:

- **deduplicates samples** by `record_id`;
- **converts units:**
  - percentages stored as fractions (0.97) become 0–100;
  - °F becomes °C;
  - decimal commas in ECG files become points;
- **is ANSI-safe:** `try_cast` / `try_to_timestamp`, so one malformed value never fails a run.

## Gold: Ready for Analysis

| Table | Grain | What's in it |
|---|---|---|
| `gold_daily_summary` | day | **The main table.** Activity, vitals, sleep, workouts, rings and calendar features in one wide row per day |
| `gold_hourly_activity` | day × hour × metric | additive metrics, deduplicated across iPhone/Watch (`chosen_source`, `n_sources`) |
| `gold_daily_activity` | day | steps, distances, active/basal kcal, flights, exercise/stand/daylight minutes, active hours |
| `gold_hourly_heart_rate` | day × hour | HR mean/min/max, sample count |
| `gold_daily_vitals` | day | HR stats, resting HR, walking HR, HRV (SDNN), SpO₂, respiratory rate, VO₂max, wrist temperature, body mass/BMI/fat, blood pressure, gait metrics, audio exposure |
| `gold_daily_sleep` | night | in bed, asleep, deep/REM/core/awake minutes, bedtime, wake time |
| `gold_workouts` | workout | silver workout plus speed, kcal/min, GPS elevation gain, route points, max speed |
| `gold_daily_workouts` | day | number of workouts, minutes, kcal, km, a training-load proxy (minutes × avg HR / 100) |
| `gold_weekly_summary` | ISO week | weekly trends |
| `gold_ecg_summary` | ECG | classification, duration, signal statistics |

### `gold_daily_summary` Columns

| Group | Columns |
|---|---|
| Calendar | `local_date`, `year`, `month`, `day_of_week` (1 = Sunday … 7 = Saturday), `is_weekend` |
| Activity | `steps`, `walk_run_km`, `cycling_km`, `swimming_m`, `active_kcal`, `basal_kcal`, `flights_climbed`, `exercise_min`, `stand_min`, `daylight_min`, `active_hours` |
| Heart | `hr_mean`, `hr_min`, `hr_max`, `hr_samples`, `resting_hr`, `walking_hr_avg`, `hrv_sdnn_ms` |
| Respiratory and fitness | `spo2_pct`, `respiratory_rate`, `vo2max`, `wrist_temp_c` |
| Body | `body_mass_kg`, `bmi`, `body_fat_pct`, `bp_systolic`, `bp_diastolic` |
| Gait | `walking_speed_kmh`, `step_length_cm`, `walking_asymmetry_pct`, `double_support_pct`, `walking_steadiness_pct`, `physical_effort` |
| Environment | `env_audio_db`, `headphone_audio_db` |
| Sleep (the night *before* the day) | `sleep_in_bed_min`, `sleep_asleep_min`, `sleep_deep_min`, `sleep_rem_min`, `sleep_core_min`, `sleep_awake_min`, `bedtime_hours_after_18` |
| Workouts | `n_workouts`, `workout_min`, `workout_kcal`, `training_load` |
| Rings | `move_goal_kcal`, `exercise_goal_min`, `stand_goal_hours`, `move_ring_closed`, `exercise_ring_closed`, `stand_ring_closed` |

Missing measurements are `NULL`, never 0. Many metrics only exist from a certain device or iOS version on.

## ML Outputs

| Local file (`data/ml/`) | Databricks table | Content |
|---|---|---|
| `forecast_resting_hr.parquet`, `forecast_hrv_sdnn_ms.parquet` | `ml_forecast_*` | `local_date`, `actual`, `predicted`, `model` (test period) |
| `forecast_metrics.parquet` | `ml_forecast_metrics` | leaderboard per target, model and split |
| `daily_anomalies.parquet` | `ml_daily_anomalies` | `anomaly_score`, `is_anomaly`, `top_features`, deviations |
| `anomaly_episodes.parquet` | `ml_anomaly_episodes` | consecutive anomalous days merged |
| `similar_days.parquet` | `ml_similar_days` | 5 nearest neighbours per day |
| `day_clusters.parquet`, `day_cluster_profiles.parquet` | `ml_day_clusters`, `ml_day_cluster_profiles` | day type per day, 2-D embedding, profiles |
| `workout_clusters.parquet` | `ml_workout_clusters` | workout type per workout |
| `ecg_beat_anomalies.parquet`, `ecg_recordings.parquet` | `ml_ecg_*` | per-beat scores (`flag_reason` = shape / rr_interval) and per-recording HR/HRV |

## Adding a Metric

HealthKit metrics already land in `silver_records` automatically, with `metric` = the type name without
the `HKQuantityTypeIdentifier` prefix. To surface one in gold:

1. Add a column to `gold_daily_vitals.sql` (or `gold_daily_activity`) and to `gold_daily_summary.sql`.
2. Run `make pipeline` and `make test-spark` (the parity test).
3. Optionally add a plausibility check in `quality.py` and a chart in the app.

[CONTRIBUTING](../CONTRIBUTING.md) describes the workflow.
