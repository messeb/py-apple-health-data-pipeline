-- Wide one-row-per-day table: the main input for the app, the dashboard and ML.
-- Sleep is attached to the day *after* the night (sleep before the day).
WITH days AS (
  SELECT DISTINCT local_date FROM silver_records
)
SELECT
  d.local_date,
  year(d.local_date) AS year,
  month(d.local_date) AS month,
  dayofweek(d.local_date) AS day_of_week,
  dayofweek(d.local_date) IN (1, 7) AS is_weekend,
  a.steps, a.walk_run_km, a.cycling_km, a.swimming_m, a.active_kcal, a.basal_kcal,
  a.flights_climbed, a.exercise_min, a.stand_min, a.daylight_min, a.active_hours,
  v.hr_mean, v.hr_min, v.hr_max, v.hr_samples, v.resting_hr, v.walking_hr_avg, v.hrv_sdnn_ms,
  v.spo2_pct, v.respiratory_rate, v.vo2max, v.wrist_temp_c,
  v.body_mass_kg, v.bmi, v.body_fat_pct, v.bp_systolic, v.bp_diastolic,
  v.walking_speed_kmh, v.step_length_cm, v.walking_asymmetry_pct, v.double_support_pct,
  v.walking_steadiness_pct, v.physical_effort, v.env_audio_db, v.headphone_audio_db,
  s.in_bed_min AS sleep_in_bed_min, s.asleep_min AS sleep_asleep_min, s.deep_min AS sleep_deep_min,
  s.rem_min AS sleep_rem_min, s.core_min AS sleep_core_min, s.awake_min AS sleep_awake_min,
  s.bedtime_hours_after_18,
  coalesce(w.n_workouts, 0) AS n_workouts,
  coalesce(w.workout_min, 0) AS workout_min,
  coalesce(w.workout_kcal, 0) AS workout_kcal,
  coalesce(w.training_load, 0) AS training_load,
  r.move_goal_kcal, r.exercise_goal_min, r.stand_goal_hours,
  r.move_ring_closed, r.exercise_ring_closed, r.stand_ring_closed
FROM days d
LEFT JOIN gold_daily_activity a ON d.local_date = a.local_date
LEFT JOIN gold_daily_vitals v ON d.local_date = v.local_date
LEFT JOIN gold_daily_sleep s ON d.local_date = date_add(s.night_date, 1)
LEFT JOIN gold_daily_workouts w ON d.local_date = w.local_date
LEFT JOIN silver_activity_summaries r ON d.local_date = r.local_date
