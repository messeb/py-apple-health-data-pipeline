-- One row per Apple Watch ECG recording. The person's name and birth date
-- are already dropped in bronze.
SELECT
  ecg_id,
  try_cast(substring(get_json_object(header_json, '$.Recorded Date'), 1, 19) AS TIMESTAMP_NTZ) AS recorded_wall,
  get_json_object(header_json, '$.Classification') AS classification,
  nullif(get_json_object(header_json, '$.Symptoms'), '') AS symptoms,
  get_json_object(header_json, '$.Device') AS device,
  get_json_object(header_json, '$.Software Version') AS software_version,
  -- Locale-formatted ("513,297 hertz") -> 513.297
  try_cast(replace(regexp_extract(get_json_object(header_json, '$.Sample Rate'), '^([0-9.,]+)', 1), ',', '.') AS DOUBLE) AS sample_rate_hz,
  get_json_object(header_json, '$.Lead') AS lead,
  get_json_object(header_json, '$.Unit') AS unit,
  try_cast(get_json_object(header_json, '$.n_samples') AS INT) AS n_samples
FROM bronze_ecg_readings
