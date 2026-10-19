-- ECG recordings with signal statistics.
SELECT
  r.ecg_id,
  r.recorded_wall AS recorded_local,
  r.classification,
  r.symptoms,
  r.device,
  r.sample_rate_hz,
  r.n_samples,
  r.n_samples / nullif(r.sample_rate_hz, 0) AS duration_s,
  AVG(s.value_uv) AS mean_uv,
  STDDEV(s.value_uv) AS std_uv,
  MIN(s.value_uv) AS min_uv,
  MAX(s.value_uv) AS max_uv
FROM silver_ecg_readings r
LEFT JOIN silver_ecg_samples s ON r.ecg_id = s.ecg_id
GROUP BY r.ecg_id, r.recorded_wall, r.classification, r.symptoms, r.device, r.sample_rate_hz, r.n_samples
