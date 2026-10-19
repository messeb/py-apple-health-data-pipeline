-- Per-workout aggregates (energy, distance, heart rate, cadence, ...), typed.
SELECT DISTINCT
  workout_id,
  regexp_replace(type, '^HKQuantityTypeIdentifier', '') AS metric,
  try_cast(sum AS DOUBLE) AS sum_value,
  try_cast(average AS DOUBLE) AS avg_value,
  try_cast(minimum AS DOUBLE) AS min_value,
  try_cast(maximum AS DOUBLE) AS max_value,
  unit,
  CASE unit
    WHEN 'km' THEN try_cast(sum AS DOUBLE)
    WHEN 'm' THEN try_cast(sum AS DOUBLE) / 1000
    WHEN 'mi' THEN try_cast(sum AS DOUBLE) * 1.609344
  END AS distance_km
FROM bronze_workout_statistics
