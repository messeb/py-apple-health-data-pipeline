-- Weekly trends (ISO weeks starting Monday).
SELECT
  to_date(date_trunc('WEEK', local_date)) AS week_start,
  COUNT(*) AS days_with_data,
  AVG(steps) AS avg_steps,
  SUM(walk_run_km) AS walk_run_km,
  SUM(cycling_km) AS cycling_km,
  AVG(active_kcal) AS avg_active_kcal,
  SUM(exercise_min) AS exercise_min,
  SUM(n_workouts) AS n_workouts,
  SUM(workout_min) AS workout_min,
  SUM(training_load) AS training_load,
  AVG(resting_hr) AS avg_resting_hr,
  AVG(hrv_sdnn_ms) AS avg_hrv_sdnn_ms,
  AVG(vo2max) AS avg_vo2max,
  AVG(sleep_asleep_min) AS avg_sleep_min,
  AVG(body_mass_kg) AS avg_body_mass_kg,
  SUM(CASE WHEN move_ring_closed THEN 1 ELSE 0 END) AS move_rings_closed
FROM gold_daily_summary
GROUP BY to_date(date_trunc('WEEK', local_date))
