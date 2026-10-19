-- One row per day with deduplicated activity totals.
SELECT
  local_date,
  SUM(CASE WHEN metric = 'StepCount' THEN value END) AS steps,
  SUM(CASE WHEN metric = 'DistanceWalkingRunning' THEN value END) AS walk_run_km,
  SUM(CASE WHEN metric = 'DistanceCycling' THEN value END) AS cycling_km,
  SUM(CASE WHEN metric = 'DistanceSwimming' THEN value END) AS swimming_m,
  SUM(CASE WHEN metric = 'ActiveEnergyBurned' THEN value END) AS active_kcal,
  SUM(CASE WHEN metric = 'BasalEnergyBurned' THEN value END) AS basal_kcal,
  SUM(CASE WHEN metric = 'FlightsClimbed' THEN value END) AS flights_climbed,
  SUM(CASE WHEN metric = 'AppleExerciseTime' THEN value END) AS exercise_min,
  SUM(CASE WHEN metric = 'AppleStandTime' THEN value END) AS stand_min,
  SUM(CASE WHEN metric = 'TimeInDaylight' THEN value END) AS daylight_min,
  COUNT(DISTINCT CASE WHEN metric = 'StepCount' AND value > 0 THEN local_hour END) AS active_hours
FROM gold_hourly_activity
GROUP BY local_date
