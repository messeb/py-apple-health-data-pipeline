import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui

ui.page_header("Overview", "How the selected period compares with the one before it")
p = ui.period()
cur, prev, wide = ui.daily_frames(p)
pal = charts.palette()

if cur.empty:
    ui.empty("No days with data in this period. Pick a wider range in the sidebar.")
    st.stop()


def rate(df: pd.DataFrame, col: str) -> float | None:
    s = df[col].dropna().astype(float)
    return float(s.mean() * 100) if len(s) else None


# Hero tiles: one number each, delta vs the previous period of equal length.
TILES = [
    # label, column, fmt, unit, lower_is_better, help
    ("Steps / day", "steps", ",.0f", "", False, "Mean daily step count (deduplicated across iPhone and Watch)"),
    ("Active energy / day", "active_kcal", ",.0f", "kcal", False, None),
    ("Exercise / day", "exercise_min", ",.0f", "min", False, "Apple exercise minutes"),
    ("Resting heart rate", "resting_hr", ",.1f", "bpm", True, "Lower usually means better cardio fitness"),
    ("HRV (SDNN)", "hrv_sdnn_ms", ",.0f", "ms", False, "Higher usually means better recovery"),
]
cols = st.columns(len(TILES))
for col, (label, c, fmt, unit, lower, help_) in zip(cols, TILES, strict=False):
    with col:
        ui.kpi(label, ui.mean(cur, c), ui.mean(prev, c), fmt, unit, lower, help_, ui.weekly_spark(cur, c))

cols = st.columns(5)
with cols[0]:
    ui.kpi("Workouts", float(cur["n_workouts"].sum()), float(prev["n_workouts"].sum()) or None, ",.0f")
with cols[1]:
    ui.kpi(
        "Workout time / week",
        cur["workout_min"].sum() / p.days * 7,
        (prev["workout_min"].sum() / p.days * 7) or None,
        ",.0f",
        "min",
    )
with cols[2]:
    ui.kpi(
        "Move ring closed",
        rate(cur, "move_ring_closed"),
        rate(prev, "move_ring_closed"),
        ",.0f",
        "%",
        help="Share of days with an activity summary where the move goal was reached",
    )
with cols[3]:
    ui.kpi(
        "VO₂max",
        ui.mean(cur, "vo2max"),
        ui.mean(prev, "vo2max"),
        ",.1f",
        "",
        help="ml/kg·min, estimated by the Watch during outdoor walks and runs; sparse",
    )
with cols[4]:
    ui.kpi("Walking speed", ui.mean(cur, "walking_speed_kmh"), ui.mean(prev, "walking_speed_kmh"), ",.2f", "km/h")

# What changed: the largest relative moves, in words.
CHANGE_METRICS = {
    "steps": ("daily steps", False),
    "active_kcal": ("active energy", False),
    "exercise_min": ("exercise minutes", False),
    "resting_hr": ("resting heart rate", True),
    "hrv_sdnn_ms": ("HRV", False),
    "flights_climbed": ("flights climbed", False),
    "daylight_min": ("time in daylight", False),
    "walking_speed_kmh": ("walking speed", False),
    "workout_min": ("workout time", False),
}
moves = []
for c, (name, lower) in CHANGE_METRICS.items():
    a, b = ui.mean(cur, c), ui.mean(prev, c)
    if a is not None and b and cur[c].notna().sum() >= 5 and prev[c].notna().sum() >= 5:
        rel = (a - b) / abs(b)
        moves.append((abs(rel), rel, name, lower))
moves.sort(reverse=True)
if moves:
    lines = []
    for _, rel, name, lower in moves[:4]:
        good = (rel < 0) if lower else (rel > 0)
        icon = ":material/trending_up:" if rel > 0 else ":material/trending_down:"
        verdict = "better" if good else "worse"
        if abs(rel) < 0.03:
            verdict, icon = "about the same", ":material/trending_flat:"
        lines.append(f"- {icon} **{name.capitalize()}** {rel:+.0%} vs previous period ({verdict})")
    with st.container(border=True):
        st.markdown("**Biggest changes**\n" + "\n".join(lines))

st.subheader("Trends")
st.caption("Dots are single days; lines are 7- and 28-day rolling means.")
charts.show(charts.trend(wide, "steps", "steps", clip=(p.start, p.end)), key="t_steps")
left, right = st.columns(2)
with left:
    st.markdown("**Resting heart rate**")
    charts.show(charts.trend(wide, "resting_hr", "bpm", fmt=".1f", clip=(p.start, p.end), height=260), key="t_rhr")
with right:
    st.markdown("**Heart rate variability (SDNN)**")
    charts.show(charts.trend(wide, "hrv_sdnn_ms", "ms", clip=(p.start, p.end), height=260), key="t_hrv")

# Year by year, independent of the period filter.
st.subheader("Year by year")
st.caption("All time, independent of the period filter. Partial first and current years are marked.")
lo, hi = data.date_bounds()
alltime = data.daily_summary(lo, hi)
yearly = (
    alltime.groupby("year")
    .agg(
        days=("local_date", "count"),
        steps=("steps", "mean"),
        resting_hr=("resting_hr", "mean"),
        exercise_min=("exercise_min", "mean"),
        workouts=("n_workouts", "sum"),
    )
    .reset_index()
)
yearly["partial"] = yearly["days"] < 330


def year_bars(col: str, unit: str, fmt: str) -> go.Figure:
    d = yearly.dropna(subset=[col])
    colors = [pal.neutral if part else pal.series[0] for part in d["partial"]]
    fig = go.Figure(
        go.Bar(
            x=d["year"].astype(str),
            y=d[col],
            marker=dict(color=colors, cornerradius=4),
            customdata=d["days"],
            hovertemplate=f"%{{x}}: %{{y:{fmt}}} {unit}<br>%{{customdata}} days<extra></extra>",
        )
    )
    charts.style(fig, height=240, legend=False, pal=pal)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(title_text=unit)
    return fig


c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Steps / day**")
    charts.show(year_bars("steps", "steps", ",.0f"), key="y_steps")
with c2:
    st.markdown("**Resting heart rate**")
    fig = year_bars("resting_hr", "bpm", ".1f")
    d = yearly.dropna(subset=["resting_hr"])
    if not d.empty:
        fig.update_yaxes(range=[d["resting_hr"].min() - 5, d["resting_hr"].max() + 3])
    charts.show(fig, key="y_rhr")
with c3:
    st.markdown("**Workouts**")
    charts.show(year_bars("workouts", "workouts", ",.0f"), key="y_workouts")

with st.expander("Data coverage: share of days with each metric, per year"):
    cov = data.coverage()
    labels = {
        "steps": "Steps",
        "active_kcal": "Active energy",
        "exercise_min": "Exercise",
        "hr_mean": "Heart rate",
        "resting_hr": "Resting HR",
        "hrv_sdnn_ms": "HRV",
        "vo2max": "VO₂max",
        "spo2_pct": "SpO₂",
        "body_mass_kg": "Body mass",
        "walking_speed_kmh": "Walking speed",
        "sleep_in_bed_min": "In bed",
        "sleep_asleep_min": "Asleep",
        "move_goal_kcal": "Activity rings",
    }
    share = cov[list(labels)].div(cov["days"], axis=0) * 100
    fig = go.Figure(
        go.Heatmap(
            z=share.T.values,
            x=cov["year"].astype(str),
            y=list(labels.values()),
            zmin=0,
            zmax=100,
            colorscale=charts.seq_scale(pal),
            xgap=2,
            ygap=2,
            colorbar=dict(
                title=dict(text="% days", side="right"), outlinewidth=0, thickness=10, tickfont=dict(color=pal.muted)
            ),
            hovertemplate="%{y} · %{x}: %{z:.0f}% of days<extra></extra>",
        )
    )
    charts.style(fig, height=420, legend=False, pal=pal)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    table = share.round(0).assign(year=cov["year"]).rename(columns=labels)
    charts.show(fig, key="coverage")
    st.dataframe(table[["year", *labels.values()]], hide_index=True)
