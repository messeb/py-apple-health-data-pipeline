import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui

ui.page_header("Workouts", "Training volume, intensity and routes")
p = ui.period()
pal = charts.palette()


def num(value, fmt: str, unit: str) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:{fmt}} {unit}"


def pretty(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name)).replace("High Intensity Interval Training", "HIIT")


w_all = data.workouts(p.start, p.end)
w_prev = data.workouts(p.previous.start, p.previous.end)
if w_all.empty:
    ui.empty("No workouts in this period.")
    st.stop()
w_all["activity_name"] = w_all["activity"].map(pretty)
w_prev["activity_name"] = w_prev["activity"].map(pretty)

ranked = w_all.groupby("activity_name")["duration_min"].sum().sort_values(ascending=False)
picked = st.multiselect(
    "Activity types", list(ranked.index), default=list(ranked.index), key="w_types", placeholder="All activity types"
)
picked = picked or list(ranked.index)
w = w_all[w_all["activity_name"].isin(picked)]
wp = w_prev[w_prev["activity_name"].isin(picked)]

cols = st.columns(5)
with cols[0]:
    ui.kpi("Workouts", float(len(w)), float(len(wp)) or None)
with cols[1]:
    ui.kpi("Total time", w["duration_min"].sum() / 60, (wp["duration_min"].sum() / 60) or None, ",.1f", "h")
with cols[2]:
    ui.kpi("Distance", w["distance_km"].sum(), wp["distance_km"].sum() or None, ",.0f", "km")
with cols[3]:
    ui.kpi("Active energy", w["active_kcal"].sum(), wp["active_kcal"].sum() or None, ",.0f", "kcal")
with cols[4]:
    ui.kpi(
        "Avg heart rate",
        ui.mean(w, "hr_avg"),
        ui.mean(wp, "hr_avg"),
        ",.0f",
        "bpm",
        help="Mean of per-workout average heart rate",
    )

# Stable color per activity: slot order follows all-time volume in the period, top 4 + Other.
top = list(ranked.index[:4])
cmap = charts.color_map([*top, "Other"], pal)

left, right = st.columns([1, 2])
with left:
    st.markdown("**Time by activity**")
    by = (
        w.groupby("activity_name")
        .agg(hours=("duration_min", lambda s: s.sum() / 60), n=("workout_id", "count"))
        .sort_values("hours")
        .reset_index()
    )
    fig = go.Figure(
        go.Bar(
            y=by["activity_name"],
            x=by["hours"],
            orientation="h",
            customdata=by["n"],
            marker=dict(color=pal.series[0], cornerradius=4),
            hovertemplate="%{y}: %{x:,.1f} h · %{customdata} workouts<extra></extra>",
        )
    )
    charts.style(fig, height=max(240, 26 * len(by) + 60), legend=False, pal=pal)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="hours", showgrid=True)
    fig.update_yaxes(showgrid=False)
    charts.show(fig, table=by.round(1), key="w_by")
with right:
    st.markdown("**Monthly volume**")
    m = w.assign(
        group=w["activity_name"].where(w["activity_name"].isin(top), "Other"),
        month=w["local_date"].dt.to_period("M").dt.to_timestamp(),
    )
    vol = m.pivot_table(index="month", columns="group", values="duration_min", aggfunc="sum").fillna(0) / 60
    fig = go.Figure()
    for g in [g for g in [*top, "Other"] if g in vol.columns]:
        fig.add_bar(
            x=vol.index,
            y=vol[g],
            name=g,
            marker=dict(color=cmap[g], line=dict(color=pal.surface, width=1)),
            hovertemplate=f"{g}: %{{y:,.1f}} h<extra></extra>",
        )
    charts.style(fig, height=max(240, 26 * len(by) + 60), pal=pal).update_layout(barmode="stack")
    fig.update_yaxes(title_text="hours")
    charts.show(fig, table=vol.round(1).reset_index(), key="w_month")

# ---------------------------------------------------------------- intensity
st.subheader("Intensity")
hr_ref = data.workout_hr_reference()
left, right = st.columns(2)
with left:
    st.markdown("**Heart-rate zones (proxy)**")
    z = w.dropna(subset=["hr_avg"])
    if z.empty or not hr_ref:
        ui.empty("No workouts with heart-rate data.")
    else:
        bins = [0, 0.6, 0.7, 0.8, 0.9, 10]
        labels = ["Z1 <60%", "Z2 60–70%", "Z3 70–80%", "Z4 80–90%", "Z5 ≥90%"]
        z = z.assign(zone=pd.cut(z["hr_avg"] / hr_ref, bins, labels=labels, right=False))
        zones = z.groupby("zone", observed=False)["duration_min"].sum().reindex(labels).fillna(0) / 60
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=zones.values,
                marker=dict(color=charts.ordinal(5, pal), cornerradius=4),
                hovertemplate="%{x}: %{y:,.1f} h<extra></extra>",
            )
        )
        charts.style(fig, height=280, legend=False, pal=pal).update_layout(hovermode="closest")
        fig.update_yaxes(title_text="hours")
        charts.show(fig, table=zones.round(1).rename("hours").reset_index(), key="w_zones")
        st.caption(
            f"Each workout's full duration is assigned to the zone of its *average* heart rate, "
            f"relative to an HRmax estimate of {hr_ref:.0f} bpm (98th percentile of all workout maxima). "
            f"A coarse proxy: per-second zone time is not in the export."
        )
with right:
    st.markdown("**Duration vs average heart rate**")
    s = w.dropna(subset=["hr_avg", "duration_min"])
    s = s.assign(group=s["activity_name"].where(s["activity_name"].isin(top), "Other"))
    fig = go.Figure()
    for g in [g for g in [*top, "Other"] if g in set(s["group"])]:
        d = s[s["group"] == g]
        fig.add_scatter(
            x=d["duration_min"],
            y=d["hr_avg"],
            mode="markers",
            name=g,
            marker=dict(size=8, color=cmap[g], opacity=0.8, line=dict(color=pal.surface, width=1)),
            customdata=np.stack([d["activity_name"], d["local_date"].dt.strftime("%d %b %Y")], axis=1),
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{x:.0f} min · %{y:.0f} bpm<extra></extra>",
        )
    charts.style(fig, height=280, pal=pal).update_layout(hovermode="closest")
    fig.update_xaxes(title_text="minutes", showgrid=True)
    fig.update_yaxes(title_text="avg bpm")
    charts.show(fig, key="w_scatter")

# ---------------------------------------------------------------- table
st.subheader("All workouts")
table = w[
    [
        "start_local",
        "activity_name",
        "duration_min",
        "distance_km",
        "avg_speed_kmh",
        "active_kcal",
        "hr_avg",
        "hr_max",
        "elevation_gain_m",
        "weather_temp_c",
        "is_indoor",
        "route_file",
    ]
].copy()
table["has_route"] = table["route_file"].notna()
st.dataframe(
    table.drop(columns="route_file"),
    hide_index=True,
    height=320,
    column_config={
        "start_local": st.column_config.DatetimeColumn("Start", format="DD MMM YYYY, HH:mm"),
        "activity_name": "Activity",
        "duration_min": st.column_config.NumberColumn("Minutes", format="%.0f"),
        "distance_km": st.column_config.NumberColumn("km", format="%.1f"),
        "avg_speed_kmh": st.column_config.NumberColumn("km/h", format="%.1f"),
        "active_kcal": st.column_config.NumberColumn("kcal", format="%.0f"),
        "hr_avg": st.column_config.NumberColumn("Avg HR", format="%.0f"),
        "hr_max": st.column_config.NumberColumn("Max HR", format="%.0f"),
        "elevation_gain_m": st.column_config.NumberColumn("Climb (m)", format="%.0f"),
        "weather_temp_c": st.column_config.NumberColumn("°C", format="%.0f"),
        "is_indoor": st.column_config.CheckboxColumn("Indoor"),
        "has_route": st.column_config.CheckboxColumn("GPS"),
    },
)

# ---------------------------------------------------------------- route map
st.subheader("Route")
routed = w.dropna(subset=["route_file"]).sort_values("start_local", ascending=False)
if routed.empty:
    ui.empty("No GPS routes among these workouts.")
    st.stop()
labels = {
    r.route_file: f"{r.start_local:%d %b %Y %H:%M} · {r.activity_name} · {num(r.distance_km, '.1f', 'km')}"
    for r in routed.itertuples()
}
by_label = {v: k for k, v in labels.items()}
route_file = by_label[st.selectbox("Workout", list(by_label), key="w_route")]
pts = data.route_points(route_file)
if pts.empty:
    ui.empty("This route has no usable GPS points.")
    st.stop()
if len(pts) > 4000:  # plenty for a line at map zoom levels
    pts = pts.iloc[:: math.ceil(len(pts) / 4000)]

lat_span = pts["lat"].max() - pts["lat"].min()
lon_span = (pts["lon"].max() - pts["lon"].min()) * math.cos(math.radians(pts["lat"].mean()))
zoom = max(2.0, min(16.0, math.log2(360 / max(lat_span, lon_span, 1e-4)) - 1.2))
fig = go.Figure()
fig.add_trace(
    go.Scattermap(
        lat=pts["lat"],
        lon=pts["lon"],
        mode="lines",
        name="Route",
        line=dict(width=4, color=pal.series[0]),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scattermap(
        lat=[pts["lat"].iloc[0], pts["lat"].iloc[-1]],
        lon=[pts["lon"].iloc[0], pts["lon"].iloc[-1]],
        mode="markers+text",
        text=["Start", "Finish"],
        textposition="top right",
        marker=dict(size=12, color=[pal.series[2], pal.series[1]]),
        textfont=dict(color="#0b0b0b", size=12),
        hoverinfo="text",
        name="",
    )
)
fig.update_layout(
    map=dict(style="open-street-map", center=dict(lat=pts["lat"].mean(), lon=pts["lon"].mean()), zoom=zoom),
    height=460,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
)
left, right = st.columns([3, 2])
with left:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False, "scrollZoom": True}, key="w_map")
with right:
    sel = routed[routed["route_file"] == route_file].iloc[0]
    a, b = st.columns(2)
    a.metric("Distance", num(sel["distance_km"], ".2f", "km"), border=True)
    b.metric("Duration", num(sel["duration_min"], ".0f", "min"), border=True)
    a.metric("Avg speed", num(sel["avg_speed_kmh"], ".1f", "km/h"), border=True)
    b.metric("Climb", num(sel["elevation_gain_m"], ".0f", "m"), border=True)
    if pts["elevation_m"].notna().any():
        t = (pts["point_local"] - pts["point_local"].iloc[0]).dt.total_seconds() / 60
        ef = go.Figure(
            go.Scatter(
                x=t,
                y=pts["elevation_m"],
                mode="lines",
                fill="tozeroy",
                line=dict(color=pal.series[0], width=2),
                fillcolor="rgba(42,120,214,0.12)" if pal.mode == "light" else "rgba(57,135,229,0.18)",
                hovertemplate="%{x:.0f} min: %{y:.0f} m<extra></extra>",
            )
        )
        charts.style(ef, height=200, legend=False, pal=pal)
        lo, hi = pts["elevation_m"].min(), pts["elevation_m"].max()
        ef.update_yaxes(title_text="elevation (m)", range=[lo - 0.1 * (hi - lo + 10), hi + 0.1 * (hi - lo + 10)])
        ef.update_xaxes(title_text="minutes")
        charts.show(ef, key="w_elev")
