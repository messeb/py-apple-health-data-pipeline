-- Typed workouts. Newer exports carry distance/energy only in
-- WorkoutStatistics, so these are joined in from silver_workout_statistics.
WITH w AS (
  SELECT
    workout_id,
    regexp_replace(workout_activity_type, '^HKWorkoutActivityType', '') AS activity,
    source_name,
    try_to_timestamp(substring(start_date, 1, 19)) AS start_wall,
    try_to_timestamp(substring(end_date, 1, 19)) AS end_wall,
    (CASE WHEN substring(start_date, 21, 1) = '-' THEN -1 ELSE 1 END)
      * (try_cast(substring(start_date, 22, 2) AS INT) * 60 + try_cast(substring(start_date, 24, 2) AS INT)) AS offset_min,
    CASE duration_unit
      WHEN 'min' THEN try_cast(duration AS DOUBLE)
      WHEN 's' THEN try_cast(duration AS DOUBLE) / 60
      WHEN 'hr' THEN try_cast(duration AS DOUBLE) * 60
    END AS duration_min,
    CASE total_distance_unit
      WHEN 'km' THEN try_cast(total_distance AS DOUBLE)
      WHEN 'm' THEN try_cast(total_distance AS DOUBLE) / 1000
      WHEN 'mi' THEN try_cast(total_distance AS DOUBLE) * 1.609344
    END AS attr_distance_km,
    CASE total_energy_burned_unit
      WHEN 'kcal' THEN try_cast(total_energy_burned AS DOUBLE)
      WHEN 'kJ' THEN try_cast(total_energy_burned AS DOUBLE) / 4.184
    END AS attr_energy_kcal,
    get_json_object(metadata_json, '$.HKIndoorWorkout') = '1' AS is_indoor,
    try_cast(regexp_extract(get_json_object(metadata_json, '$.HKAverageMETs'), '^([0-9.]+)', 1) AS DOUBLE) AS avg_mets,
    try_cast(regexp_extract(get_json_object(metadata_json, '$.HKWeatherHumidity'), '^([0-9.]+)', 1) AS DOUBLE) / 100 AS weather_humidity_pct,
    CASE
      WHEN get_json_object(metadata_json, '$.HKWeatherTemperature') LIKE '%degF'
        THEN (try_cast(regexp_extract(get_json_object(metadata_json, '$.HKWeatherTemperature'), '^([0-9.-]+)', 1) AS DOUBLE) - 32) * 5 / 9
      ELSE try_cast(regexp_extract(get_json_object(metadata_json, '$.HKWeatherTemperature'), '^([0-9.-]+)', 1) AS DOUBLE)
    END AS weather_temp_c,
    try_cast(regexp_extract(get_json_object(metadata_json, '$.HKElevationAscended'), '^([0-9.]+)', 1) AS DOUBLE) / 100 AS elevation_ascended_m,
    route_file,
    metadata_json
  FROM bronze_workouts
),
stats AS (
  SELECT
    workout_id,
    SUM(CASE WHEN metric = 'ActiveEnergyBurned' THEN sum_value END) AS active_kcal,
    SUM(CASE WHEN metric = 'BasalEnergyBurned' THEN sum_value END) AS basal_kcal,
    SUM(CASE WHEN metric LIKE 'Distance%' THEN distance_km END) AS stat_distance_km,
    MAX(CASE WHEN metric = 'HeartRate' THEN avg_value END) AS hr_avg,
    MAX(CASE WHEN metric = 'HeartRate' THEN min_value END) AS hr_min,
    MAX(CASE WHEN metric = 'HeartRate' THEN max_value END) AS hr_max,
    MAX(CASE WHEN metric = 'CyclingCadence' THEN avg_value END) AS cadence_avg,
    MAX(CASE WHEN metric = 'SwimmingStrokeCount' THEN sum_value END) AS swim_strokes
  FROM silver_workout_statistics
  GROUP BY workout_id
)
SELECT
  w.workout_id,
  w.activity,
  w.source_name,
  CAST(from_utc_timestamp(timestampadd(MINUTE, -w.offset_min, w.start_wall), '{{TIMEZONE}}') AS TIMESTAMP_NTZ) AS start_local,
  CAST(from_utc_timestamp(timestampadd(MINUTE, -w.offset_min, w.end_wall), '{{TIMEZONE}}') AS TIMESTAMP_NTZ) AS end_local,
  to_date(from_utc_timestamp(timestampadd(MINUTE, -w.offset_min, w.start_wall), '{{TIMEZONE}}')) AS local_date,
  w.duration_min,
  coalesce(w.attr_distance_km, s.stat_distance_km) AS distance_km,
  coalesce(w.attr_energy_kcal, s.active_kcal) AS active_kcal,
  s.basal_kcal,
  s.hr_avg,
  s.hr_min,
  s.hr_max,
  s.cadence_avg,
  s.swim_strokes,
  w.is_indoor,
  w.avg_mets,
  w.weather_temp_c,
  w.weather_humidity_pct,
  w.elevation_ascended_m,
  w.route_file,
  w.metadata_json
FROM w
LEFT JOIN stats s ON w.workout_id = s.workout_id
