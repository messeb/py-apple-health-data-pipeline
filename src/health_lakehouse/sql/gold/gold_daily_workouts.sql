-- Workout load per day.
SELECT
  local_date,
  COUNT(*) AS n_workouts,
  SUM(duration_min) AS workout_min,
  SUM(active_kcal) AS workout_kcal,
  SUM(distance_km) AS workout_km,
  MAX(hr_max) AS workout_hr_max,
  SUM(CASE WHEN activity = 'Cycling' THEN duration_min ELSE 0 END) AS cycling_workout_min,
  SUM(CASE WHEN activity IN ('Walking', 'Hiking', 'Running') THEN duration_min ELSE 0 END) AS walk_run_workout_min,
  SUM(CASE WHEN activity IN ('TraditionalStrengthTraining', 'FunctionalStrengthTraining', 'Climbing') THEN duration_min ELSE 0 END) AS strength_workout_min,
  -- Simple training impulse proxy: minutes x average heart rate / 100
  SUM(duration_min * coalesce(hr_avg, 100) / 100) AS training_load
FROM gold_workouts
GROUP BY local_date
