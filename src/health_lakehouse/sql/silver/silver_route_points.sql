-- GPS track points of outdoor workouts (GPX timestamps are UTC).
SELECT
  route_file,
  point_index,
  try_cast(lat AS DOUBLE) AS lat,
  try_cast(lon AS DOUBLE) AS lon,
  try_cast(ele AS DOUBLE) AS elevation_m,
  CAST(from_utc_timestamp(try_to_timestamp(substring(replace(time, 'T', ' '), 1, 19)), '{{TIMEZONE}}') AS TIMESTAMP_NTZ) AS point_local,
  try_cast(speed AS DOUBLE) AS speed_mps,
  try_cast(course AS DOUBLE) AS course_deg,
  try_cast(h_acc AS DOUBLE) AS h_accuracy_m,
  try_cast(v_acc AS DOUBLE) AS v_accuracy_m
FROM bronze_route_points
