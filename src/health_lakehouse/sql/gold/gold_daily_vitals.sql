-- Daily cardio, respiratory, body and gait metrics.
SELECT
  local_date,
  AVG(CASE WHEN metric = 'HeartRate' AND value_num BETWEEN 25 AND 250 THEN value_num END) AS hr_mean,
  MIN(CASE WHEN metric = 'HeartRate' AND value_num BETWEEN 25 AND 250 THEN value_num END) AS hr_min,
  MAX(CASE WHEN metric = 'HeartRate' AND value_num BETWEEN 25 AND 250 THEN value_num END) AS hr_max,
  COUNT(CASE WHEN metric = 'HeartRate' THEN 1 END) AS hr_samples,
  AVG(CASE WHEN metric = 'RestingHeartRate' THEN value_num END) AS resting_hr,
  AVG(CASE WHEN metric = 'WalkingHeartRateAverage' THEN value_num END) AS walking_hr_avg,
  AVG(CASE WHEN metric = 'HeartRateVariabilitySDNN' THEN value_num END) AS hrv_sdnn_ms,
  AVG(CASE WHEN metric = 'HeartRateRecoveryOneMinute' THEN value_num END) AS hr_recovery_1min,
  AVG(CASE WHEN metric = 'OxygenSaturation' THEN value_num END) AS spo2_pct,
  AVG(CASE WHEN metric = 'RespiratoryRate' THEN value_num END) AS respiratory_rate,
  AVG(CASE WHEN metric = 'VO2Max' THEN value_num END) AS vo2max,
  AVG(CASE WHEN metric = 'AppleSleepingWristTemperature' THEN value_num END) AS wrist_temp_c,
  max_by(CASE WHEN metric = 'BodyMass' THEN value_num END, CASE WHEN metric = 'BodyMass' THEN start_utc END) AS body_mass_kg,
  max_by(CASE WHEN metric = 'BodyMassIndex' THEN value_num END, CASE WHEN metric = 'BodyMassIndex' THEN start_utc END) AS bmi,
  max_by(CASE WHEN metric = 'BodyFatPercentage' THEN value_num END, CASE WHEN metric = 'BodyFatPercentage' THEN start_utc END) AS body_fat_pct,
  AVG(CASE WHEN metric = 'BloodPressureSystolic' THEN value_num END) AS bp_systolic,
  AVG(CASE WHEN metric = 'BloodPressureDiastolic' THEN value_num END) AS bp_diastolic,
  AVG(CASE WHEN metric = 'WalkingSpeed' THEN value_num END) AS walking_speed_kmh,
  AVG(CASE WHEN metric = 'WalkingStepLength' THEN value_num END) AS step_length_cm,
  AVG(CASE WHEN metric = 'WalkingAsymmetryPercentage' THEN value_num END) AS walking_asymmetry_pct,
  AVG(CASE WHEN metric = 'WalkingDoubleSupportPercentage' THEN value_num END) AS double_support_pct,
  AVG(CASE WHEN metric = 'AppleWalkingSteadiness' THEN value_num END) AS walking_steadiness_pct,
  AVG(CASE WHEN metric = 'PhysicalEffort' THEN value_num END) AS physical_effort,
  AVG(CASE WHEN metric = 'EnvironmentalAudioExposure' THEN value_num END) AS env_audio_db,
  AVG(CASE WHEN metric = 'HeadphoneAudioExposure' THEN value_num END) AS headphone_audio_db
FROM silver_records
WHERE kind = 'quantity'
GROUP BY local_date
