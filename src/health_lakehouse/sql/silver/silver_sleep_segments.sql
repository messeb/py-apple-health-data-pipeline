-- Sleep analysis intervals with normalised stage names. A night is labelled
-- with the date of the evening it started (start shifted back by 12 hours).
SELECT
  record_id,
  source_name,
  device_class,
  CASE value_str
    WHEN 'HKCategoryValueSleepAnalysisInBed' THEN 'in_bed'
    WHEN 'HKCategoryValueSleepAnalysisAwake' THEN 'awake'
    WHEN 'HKCategoryValueSleepAnalysisAsleepCore' THEN 'core'
    WHEN 'HKCategoryValueSleepAnalysisAsleepDeep' THEN 'deep'
    WHEN 'HKCategoryValueSleepAnalysisAsleepREM' THEN 'rem'
    ELSE 'asleep_unspecified'
  END AS stage,
  start_local,
  end_local,
  duration_s / 60 AS duration_min,
  to_date(timestampadd(HOUR, -12, start_local)) AS night_date
FROM silver_records
WHERE metric = 'SleepAnalysis'
  AND duration_s > 0
  AND duration_s < 24 * 3600
