-- Pause/resume/lap/segment markers inside workouts.
SELECT DISTINCT
  workout_id,
  regexp_replace(type, '^HKWorkoutEventType', '') AS event_type,
  CAST(from_utc_timestamp(
    timestampadd(
      MINUTE,
      -((CASE WHEN substring(date, 21, 1) = '-' THEN -1 ELSE 1 END)
        * (try_cast(substring(date, 22, 2) AS INT) * 60 + try_cast(substring(date, 24, 2) AS INT))),
      try_to_timestamp(substring(date, 1, 19))
    ),
    '{{TIMEZONE}}'
  ) AS TIMESTAMP_NTZ) AS event_local,
  CASE duration_unit WHEN 'min' THEN try_cast(duration AS DOUBLE) WHEN 's' THEN try_cast(duration AS DOUBLE) / 60 END AS duration_min
FROM bronze_workout_events
