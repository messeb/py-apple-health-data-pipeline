-- Workout fact table enriched with GPS route metrics.
WITH pts AS (
  SELECT
    route_file,
    elevation_m,
    speed_mps,
    elevation_m - LAG(elevation_m) OVER (PARTITION BY route_file ORDER BY point_index) AS d_ele
  FROM silver_route_points
  WHERE h_accuracy_m IS NULL OR h_accuracy_m < 50
),
routes AS (
  SELECT
    route_file,
    COUNT(*) AS route_points,
    SUM(CASE WHEN d_ele > 0 THEN d_ele ELSE 0 END) AS route_elevation_gain_m,
    MAX(speed_mps) * 3.6 AS route_max_speed_kmh,
    MIN(elevation_m) AS route_min_elevation_m,
    MAX(elevation_m) AS route_max_elevation_m
  FROM pts
  GROUP BY route_file
)
SELECT
  w.workout_id,
  w.activity,
  w.source_name,
  w.local_date,
  w.start_local,
  w.end_local,
  hour(w.start_local) AS start_hour,
  w.duration_min,
  w.distance_km,
  CASE WHEN w.distance_km > 0 AND w.duration_min > 0 THEN w.distance_km / (w.duration_min / 60) END AS avg_speed_kmh,
  w.active_kcal,
  w.basal_kcal,
  CASE WHEN w.duration_min > 0 THEN w.active_kcal / w.duration_min END AS kcal_per_min,
  w.hr_avg,
  w.hr_min,
  w.hr_max,
  w.cadence_avg,
  w.swim_strokes,
  w.is_indoor,
  w.avg_mets,
  w.weather_temp_c,
  w.weather_humidity_pct,
  coalesce(w.elevation_ascended_m, r.route_elevation_gain_m) AS elevation_gain_m,
  r.route_points,
  r.route_max_speed_kmh,
  w.route_file
FROM silver_workouts w
LEFT JOIN routes r ON w.route_file = r.route_file
