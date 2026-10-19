-- HealthKit characteristics (date of birth, sex, blood type, ...) as key/value.
SELECT DISTINCT
  regexp_replace(attribute, '^HKCharacteristicTypeIdentifier', '') AS attribute,
  regexp_replace(value, '^HK(BiologicalSex|BloodType|FitzpatrickSkinType|CardioFitnessMedicationsUse)', '') AS value
FROM bronze_me
