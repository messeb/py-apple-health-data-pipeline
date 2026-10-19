-- Daily Apple Activity rings (move / exercise / stand) and goals.
SELECT
  try_cast(date_components AS DATE) AS local_date,
  try_cast(active_energy_burned AS DOUBLE) AS move_kcal,
  try_cast(active_energy_burned_goal AS DOUBLE) AS move_goal_kcal,
  try_cast(apple_exercise_time AS DOUBLE) AS exercise_min,
  try_cast(apple_exercise_time_goal AS DOUBLE) AS exercise_goal_min,
  try_cast(apple_stand_hours AS DOUBLE) AS stand_hours,
  try_cast(apple_stand_hours_goal AS DOUBLE) AS stand_goal_hours,
  try_cast(active_energy_burned AS DOUBLE) >= try_cast(active_energy_burned_goal AS DOUBLE)
    AND try_cast(active_energy_burned_goal AS DOUBLE) > 0 AS move_ring_closed,
  try_cast(apple_exercise_time AS DOUBLE) >= try_cast(apple_exercise_time_goal AS DOUBLE)
    AND try_cast(apple_exercise_time_goal AS DOUBLE) > 0 AS exercise_ring_closed,
  try_cast(apple_stand_hours AS DOUBLE) >= try_cast(apple_stand_hours_goal AS DOUBLE)
    AND try_cast(apple_stand_hours_goal AS DOUBLE) > 0 AS stand_ring_closed
FROM bronze_activity_summaries
WHERE try_cast(date_components AS DATE) >= DATE'2000-01-01'
