-- Additive activity metrics per local hour. iPhone and Apple Watch often
-- record the same steps/distance in parallel; summing all sources would
-- double count. Per hour the source with the largest total wins, which
-- closely mirrors how the Health app deduplicates.
WITH per_source AS (
  SELECT
    local_date,
    local_hour,
    metric,
    source_name,
    SUM(value_num) AS value
  FROM silver_records
  WHERE kind = 'quantity'
    AND metric IN (
      'StepCount', 'DistanceWalkingRunning', 'DistanceCycling', 'DistanceSwimming',
      'ActiveEnergyBurned', 'BasalEnergyBurned', 'FlightsClimbed',
      'AppleExerciseTime', 'AppleStandTime', 'TimeInDaylight', 'SwimmingStrokeCount'
    )
  GROUP BY local_date, local_hour, metric, source_name
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY local_date, local_hour, metric
      ORDER BY value DESC NULLS LAST, source_name
    ) AS rk,
    COUNT(*) OVER (PARTITION BY local_date, local_hour, metric) AS n_sources
  FROM per_source
)
SELECT
  local_date,
  local_hour,
  metric,
  value,
  source_name AS chosen_source,
  n_sources
FROM ranked
WHERE rk = 1
