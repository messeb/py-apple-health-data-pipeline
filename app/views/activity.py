import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui

ui.page_header("Activity", "Movement, energy and the Activity rings")
p = ui.period()
cur, prev, wide = ui.daily_frames(p)
pal = charts.palette()

if cur["steps"].notna().sum() == 0:
    ui.empty("No activity data in this period.")
    st.stop()

cols = st.columns(5)
for col, (label, c, fmt, unit) in zip(
    cols,
    [
        ("Steps / day", "steps", ",.0f", ""),
        ("Walk + run / day", "walk_run_km", ",.1f", "km"),
        ("Flights / day", "flights_climbed", ",.1f", ""),
        ("Active hours / day", "active_hours", ",.1f", "h"),
        ("Daylight / day", "daylight_min", ",.0f", "min"),
    ],
    strict=False,
):
    with col:
        ui.kpi(label, ui.mean(cur, c), ui.mean(prev, c), fmt, unit, spark=ui.weekly_spark(cur, c))

# ---------------------------------------------------------------- steps per week
st.subheader("Steps")
weeks = cur.set_index("local_date")["steps"].resample("W-MON", label="left", closed="left").mean().dropna()
by_dow = cur.groupby("day_of_week")["steps"].mean().reindex([2, 3, 4, 5, 6, 7, 1])
left, right = st.columns([2, 1])
with left:
    st.markdown("**Average steps per day, by week**")
    fig = go.Figure(
        go.Bar(
            x=weeks.index,
            y=weeks.values,
            marker=dict(color=pal.series[0], cornerradius=4),
            hovertemplate="Week of %{x|%d %b %Y}: %{y:,.0f} steps/day<extra></extra>",
        )
    )
    overall = cur["steps"].mean()
    fig.add_hline(
        y=overall,
        line=dict(color=pal.ink2, width=1),
        annotation=dict(text=f"period mean {overall:,.0f}", font=dict(color=pal.ink2, size=11)),
        annotation_position="top right",
    )
    charts.style(fig, height=280, legend=False, pal=pal).update_layout(hovermode="closest")
    charts.show(fig, table=weeks.round(0).rename("avg_steps").reset_index(), key="a_weeks")
with right:
    st.markdown("**By weekday**")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=by_dow.values,
            marker=dict(color=pal.series[0], cornerradius=4),
            hovertemplate="%{x}: %{y:,.0f} steps/day<extra></extra>",
        )
    )
    charts.style(fig, height=280, legend=False, pal=pal).update_layout(hovermode="closest")
    charts.show(fig, key="a_dow")
    wk, we = cur.loc[~cur["is_weekend"], "steps"].mean(), cur.loc[cur["is_weekend"], "steps"].mean()
    if pd.notna(wk) and pd.notna(we) and wk:
        st.caption(f"Weekends: {we:,.0f} steps/day, {(we - wk) / wk:+.0%} vs weekdays.")

# ---------------------------------------------------------------- when you move
st.subheader("When you move")
metric = (
    st.segmented_control(
        "Metric", list(data.HOURLY_METRICS), default="StepCount", format_func=data.HOURLY_METRICS.get, key="a_metric"
    )
    or "StepCount"
)
prof = data.hourly_activity_profile(metric, p.start, p.end)
if prof.empty:
    ui.empty("No hourly records for this metric in the period.")
else:
    label = data.HOURLY_METRICS[metric]
    fmt = ",.2f" if metric in ("DistanceWalkingRunning", "FlightsClimbed") else ",.0f"
    charts.show(charts.weekday_hour_heatmap(prof, "avg_value", label, fmt=fmt, fill=0), key="a_heat")
    peak = prof.loc[prof["avg_value"].idxmax()]
    day = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
    st.caption(
        f"Average per hour across the period (hours without records count as zero). "
        f"Busiest slot: {day[int(peak['day_of_week'])]} {int(peak['local_hour']):02d}:00 "
        f"with {peak['avg_value']:{fmt}}."
    )

# ---------------------------------------------------------------- distance & energy
left, right = st.columns(2)
with left:
    st.markdown("**Distance per month**")
    monthly = cur.set_index("local_date")[["walk_run_km", "cycling_km"]].resample("MS").sum(min_count=1).fillna(0)
    names = {"walk_run_km": "Walking + running", "cycling_km": "Cycling"}
    cmap = charts.color_map(list(names.values()), pal)
    fig = go.Figure()
    for c, name in names.items():
        fig.add_bar(
            x=monthly.index,
            y=monthly[c],
            name=name,
            marker=dict(color=cmap[name], line=dict(color=pal.surface, width=1)),
            hovertemplate=f"{name}: %{{y:,.0f}} km<extra></extra>",
        )
    charts.style(fig, height=300, pal=pal).update_layout(barmode="stack")
    fig.update_yaxes(title_text="km")
    charts.show(fig, table=monthly.round(1).rename(columns=names).reset_index(), key="a_dist")
with right:
    st.markdown("**Active energy vs move goal**")
    fig = charts.trend(wide, "active_kcal", "kcal", clip=(p.start, p.end), show_r7=False, height=300)
    goal = cur[["local_date", "move_goal_kcal"]].dropna()
    if not goal.empty:
        fig.add_scatter(
            x=goal["local_date"],
            y=goal["move_goal_kcal"],
            name="Move goal",
            mode="lines",
            line=dict(color=pal.ink2, width=1, shape="hv"),
            hovertemplate="%{y:,.0f} kcal<extra>Goal</extra>",
        )
    charts.show(fig, key="a_energy")

# ---------------------------------------------------------------- rings
st.subheader("Activity rings")
rings = cur.dropna(subset=["move_ring_closed"])
if rings.empty:
    ui.empty("No Activity ring summaries in this period (they start with the Apple Watch).")
else:
    names = {"move_ring_closed": "Move", "exercise_ring_closed": "Exercise", "stand_ring_closed": "Stand"}
    m = (rings.set_index("local_date")[list(names)].astype(float).resample("MS").mean() * 100).rename(columns=names)
    cmap = charts.color_map(list(names.values()), pal)
    fig = go.Figure()
    for name in names.values():
        s = m[name].dropna()
        fig.add_scatter(
            x=s.index,
            y=s.values,
            name=name,
            mode="lines+markers",
            line=dict(color=cmap[name], width=2),
            marker=dict(size=6),
            hovertemplate=f"{name}: %{{y:.0f}}%<extra></extra>",
        )
    charts.style(fig, height=300, pal=pal)
    fig.update_yaxes(title_text="% of days closed", range=[0, 105])
    charts.show(fig, table=m.round(0).reset_index(), key="a_rings")
    closed_all = rings[list(names)].fillna(False).astype(bool).all(axis=1).mean()
    st.caption(f"All three rings closed on {closed_all:.0%} of {len(rings):,} days with a ring summary.")
