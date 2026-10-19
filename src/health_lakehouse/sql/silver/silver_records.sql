-- One typed row per HealthKit sample (quantity + category), deduplicated.
-- Apple stamps every timestamp with the timezone active at *export* time, so
-- wall-clock strings are converted to UTC first and then to the configured
-- IANA timezone ({{TIMEZONE}}) to get correct local time across DST.
-- *_utc columns are instants (TIMESTAMP); *_local columns are wall-clock
-- times (TIMESTAMP_NTZ) so readers never shift them by their session timezone.
WITH dedup AS (
  SELECT
    record_id, type, source_name, source_version, device, unit, value,
    creation_date, start_date, end_date, metadata_json, correlation_id,
    -- Exact duplicates may differ only in correlation_id/metadata; the tie-breakers
    -- make the choice deterministic across engines.
    ROW_NUMBER() OVER (
      PARTITION BY record_id
      ORDER BY _ingested_at DESC, coalesce(correlation_id, '') DESC, coalesce(metadata_json, '') DESC
    ) AS rn
  FROM bronze_records
),
offsets AS (
  SELECT
    *,
    try_to_timestamp(substring(start_date, 1, 19)) AS start_wall,
    try_to_timestamp(substring(end_date, 1, 19)) AS end_wall,
    (CASE WHEN substring(start_date, 21, 1) = '-' THEN -1 ELSE 1 END)
      * (try_cast(substring(start_date, 22, 2) AS INT) * 60 + try_cast(substring(start_date, 24, 2) AS INT)) AS start_offset_min,
    (CASE WHEN substring(end_date, 21, 1) = '-' THEN -1 ELSE 1 END)
      * (try_cast(substring(end_date, 22, 2) AS INT) * 60 + try_cast(substring(end_date, 24, 2) AS INT)) AS end_offset_min
  FROM dedup
  WHERE rn = 1
),
utc AS (
  SELECT
    *,
    timestampadd(MINUTE, -start_offset_min, start_wall) AS start_utc,
    timestampadd(MINUTE, -end_offset_min, end_wall) AS end_utc
  FROM offsets
),
typed AS (
  SELECT
    record_id,
    CASE
      WHEN type LIKE 'HKQuantityTypeIdentifier%' THEN 'quantity'
      WHEN type LIKE 'HKCategoryTypeIdentifier%' THEN 'category'
      ELSE 'other'
    END AS kind,
    regexp_replace(type, '^HK(Quantity|Category|Data)Type(Identifier)?', '') AS metric,
    type AS hk_type,
    source_name,
    CASE
      WHEN lower(source_name) LIKE '%watch%' OR device LIKE '%name:Apple Watch%' THEN 'watch'
      WHEN lower(source_name) LIKE '%iphone%' OR device LIKE '%name:iPhone%' THEN 'phone'
      ELSE 'other'
    END AS device_class,
    -- HealthKit stores percentages as fractions (0.97); expose them as 0-100.
    CASE WHEN unit = '%' THEN try_cast(value AS DOUBLE) * 100 ELSE try_cast(value AS DOUBLE) END AS value_num,
    CASE WHEN try_cast(value AS DOUBLE) IS NULL THEN value END AS value_str,
    unit,
    start_utc,
    end_utc,
    CAST(from_utc_timestamp(start_utc, '{{TIMEZONE}}') AS TIMESTAMP_NTZ) AS start_local,
    CAST(from_utc_timestamp(end_utc, '{{TIMEZONE}}') AS TIMESTAMP_NTZ) AS end_local,
    timestampdiff(SECOND, start_utc, end_utc) AS duration_s,
    metadata_json,
    correlation_id
  FROM utc
)
SELECT
  *,
  to_date(start_local) AS local_date,
  hour(start_local) AS local_hour
FROM typed
