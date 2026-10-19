-- FHIR clinical records (currently immunizations), flattened.
SELECT
  record_file,
  resource_type,
  get_json_object(raw_json, '$.status') AS status,
  try_cast(substring(get_json_object(raw_json, '$.occurrenceDateTime'), 1, 10) AS DATE) AS occurrence_date,
  get_json_object(raw_json, '$.vaccineCode.coding[0].display') AS vaccine,
  get_json_object(raw_json, '$.manufacturer.display') AS manufacturer,
  get_json_object(raw_json, '$.protocolApplied[0].targetedDisease.coding[0].display') AS target_disease,
  try_cast(get_json_object(raw_json, '$.protocolApplied[0].doseNumberPositiveInt') AS INT) AS dose_number
FROM bronze_clinical_records
