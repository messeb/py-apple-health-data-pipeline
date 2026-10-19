-- ECG voltage samples in microvolts (locale decimal comma converted).
SELECT
  ecg_id,
  sample_index,
  try_cast(replace(value_raw, ',', '.') AS DOUBLE) AS value_uv
FROM bronze_ecg_samples
