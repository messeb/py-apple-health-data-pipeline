-- Heart-rate profile per local hour (basis for heatmaps and sequence models).
SELECT
  local_date,
  local_hour,
  AVG(value_num) AS hr_mean,
  MIN(value_num) AS hr_min,
  MAX(value_num) AS hr_max,
  COUNT(*) AS n_samples
FROM silver_records
WHERE metric = 'HeartRate'
  AND value_num BETWEEN 25 AND 250
GROUP BY local_date, local_hour
