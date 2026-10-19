-- One row per night. Several sources (iPhone, Watch, sleep apps) may track
-- the same night; per stage the source with the most minutes is used.
WITH per_source AS (
  SELECT night_date, stage, source_name,
         SUM(duration_min) AS minutes,
         MIN(start_local) AS first_start,
         MAX(end_local) AS last_end
  FROM silver_sleep_segments
  GROUP BY night_date, stage, source_name
),
per_stage AS (
  SELECT night_date, stage,
         MAX(minutes) AS minutes,
         MIN(first_start) AS first_start,
         MAX(last_end) AS last_end
  FROM per_source
  GROUP BY night_date, stage
)
SELECT
  night_date,
  SUM(CASE WHEN stage = 'in_bed' THEN minutes END) AS in_bed_min,
  SUM(CASE WHEN stage IN ('core', 'deep', 'rem', 'asleep_unspecified') THEN minutes END) AS asleep_min,
  SUM(CASE WHEN stage = 'deep' THEN minutes END) AS deep_min,
  SUM(CASE WHEN stage = 'rem' THEN minutes END) AS rem_min,
  SUM(CASE WHEN stage = 'core' THEN minutes END) AS core_min,
  SUM(CASE WHEN stage = 'awake' THEN minutes END) AS awake_min,
  MIN(first_start) AS bedtime_local,
  MAX(last_end) AS wake_local,
  -- Hours after 18:00 the night started (e.g. 23:30 -> 5.5); robust across midnight.
  (hour(MIN(first_start)) + minute(MIN(first_start)) / 60 + 6) % 24 AS bedtime_hours_after_18,
  COUNT(DISTINCT CASE WHEN stage IN ('core', 'deep', 'rem') THEN stage END) > 0 AS has_stages
FROM per_stage
GROUP BY night_date
