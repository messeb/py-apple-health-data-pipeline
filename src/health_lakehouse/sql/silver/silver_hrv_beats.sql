-- Instantaneous beat-to-beat BPM series attached to HRV (SDNN) samples.
-- `time` is a clock time within the HRV measurement window.
SELECT
  b.record_id,
  r.start_local AS hrv_start_local,
  r.value_num AS sdnn_ms,
  try_cast(b.bpm AS DOUBLE) AS bpm,
  b.time AS beat_time_raw
FROM bronze_hrv_instantaneous_bpm b
JOIN silver_records r ON b.record_id = r.record_id
