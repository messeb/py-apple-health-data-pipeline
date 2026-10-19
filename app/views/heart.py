import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui

ui.page_header("Heart", "Resting heart rate, variability, fitness and oxygen saturation")
p = ui.period()
cur, prev, wide = ui.daily_frames(p)
pal = charts.palette()

if cur["hr_mean"].notna().sum() == 0:
    ui.empty("No heart-rate data in this period (heart rate starts with the Apple Watch).")
    st.stop()

cols = st.columns(3) + st.columns(3)
for col, (label, c, fmt, unit, lower) in zip(
    cols,
    [
        ("Resting HR", "resting_hr", ",.1f", "bpm", True),
        ("HRV (SDNN)", "hrv_sdnn_ms", ",.0f", "ms", False),
        ("Walking HR", "walking_hr_avg", ",.0f", "bpm", True),
        ("VO₂max", "vo2max", ",.1f", "", False),
        ("SpO₂", "spo2_pct", ",.1f", "%", False),
        ("Respiratory rate", "respiratory_rate", ",.1f", "/min", True),
    ],
    strict=False,
):
    with col:
        ui.kpi(label, ui.mean(cur, c), ui.mean(prev, c), fmt, unit, lower, spark=ui.weekly_spark(cur, c))

st.subheader("Resting heart rate")
charts.show(charts.trend(wide, "resting_hr", "bpm", fmt=".1f", clip=(p.start, p.end)), key="h_rhr")

left, right = st.columns(2)
with left:
    st.markdown("**Heart rate variability (SDNN)**")
    charts.show(charts.trend(wide, "hrv_sdnn_ms", "ms", clip=(p.start, p.end), height=280), key="h_hrv")
with right:
    st.markdown("**VO₂max (cardio fitness)**")
    v = cur[["local_date", "vo2max"]].dropna()
    if v.empty:
        ui.empty("No VO₂max estimates in this period. The Watch only estimates it during outdoor walks/runs.")
    else:
        fig = go.Figure(
            go.Scatter(
                x=v["local_date"],
                y=v["vo2max"],
                mode="lines+markers",
                name="VO₂max",
                line=dict(color=pal.series[0], width=2),
                marker=dict(size=8, line=dict(color=pal.surface, width=2)),
                hovertemplate="%{x|%d %b %Y}: %{y:.1f} ml/kg·min<extra></extra>",
            )
        )
        charts.style(fig, height=280, legend=False, pal=pal)
        fig.update_yaxes(title_text="ml/kg·min", range=[v["vo2max"].min() - 0.6, v["vo2max"].max() + 0.4])
        charts.show(fig, key="h_vo2")
        st.caption(f"{len(v)} estimates in this period; lines connect consecutive measurements.")

st.subheader("Daily rhythm")
prof = data.hourly_heart_rate_profile(p.start, p.end)
if prof.empty:
    ui.empty("No hourly heart-rate samples in this period.")
else:
    charts.show(charts.weekday_hour_heatmap(prof, "hr_mean", "bpm", fmt=".0f"), key="h_heat")
    night = prof[prof["local_hour"].between(1, 5)]
    day = prof[prof["local_hour"].between(9, 20)]

    def w(d):
        return (d["hr_mean"] * d["n_samples"]).sum() / d["n_samples"].sum() if d["n_samples"].sum() else float("nan")

    st.caption(
        f"Mean heart rate per hour, weighted by sample count. 01–05 h: {w(night):.0f} bpm · "
        f"09–20 h: {w(day):.0f} bpm. Watch sampling is denser during workouts, which lifts busy hours."
    )

left, right = st.columns(2)
with left:
    st.markdown("**Does activity lower tomorrow's resting heart rate?**")
    d = cur[["local_date", "steps", "resting_hr", "hrv_sdnn_ms"]].copy().set_index("local_date").asfreq("D")
    d["next_rhr"] = d["resting_hr"].shift(-1)
    d = d.dropna(subset=["steps", "next_rhr"])
    if len(d) < 50:
        ui.empty("Needs at least 50 days with steps and next-day resting HR; widen the period.")
    else:
        d["bucket"] = pd.qcut(d["steps"], 5, labels=False, duplicates="drop")
        g = (
            d.groupby("bucket")
            .agg(
                steps_lo=("steps", "min"),
                steps_hi=("steps", "max"),
                next_rhr=("next_rhr", "mean"),
                n=("next_rhr", "size"),
            )
            .reset_index()
        )
        g["label"] = [f"{lo / 1000:.0f}–{hi / 1000:.0f}k" for lo, hi in zip(g["steps_lo"], g["steps_hi"], strict=False)]
        fig = go.Figure(
            go.Bar(
                x=g["label"],
                y=g["next_rhr"],
                marker=dict(color=charts.ordinal(len(g), pal), cornerradius=4),
                customdata=g["n"],
                hovertemplate="%{x} steps: %{y:.1f} bpm next day<br>%{customdata} days<extra></extra>",
            )
        )
        charts.style(fig, height=280, legend=False, pal=pal).update_layout(hovermode="closest")
        fig.update_xaxes(title_text="Steps that day (quintiles)")
        fig.update_yaxes(
            title_text="Next-day resting HR (bpm)", range=[g["next_rhr"].min() - 3, g["next_rhr"].max() + 1.5]
        )
        charts.show(fig, table=g[["label", "n", "next_rhr"]].round(1), key="h_next")
        st.caption("Association, not causation; days are grouped by step-count quintile.")
with right:
    st.markdown("**Blood oxygen (SpO₂), daily mean**")
    s = cur["spo2_pct"].dropna()
    if s.empty:
        ui.empty("No SpO₂ readings in this period.")
    else:
        s = s * 100 if s.max() <= 1.0 else s
        fig = go.Figure(
            go.Histogram(
                x=s,
                xbins=dict(size=0.5),
                marker=dict(color=pal.series[0], line=dict(color=pal.surface, width=2)),
                hovertemplate="%{x}%: %{y} days<extra></extra>",
            )
        )
        charts.style(fig, height=280, legend=False, pal=pal).update_layout(hovermode="closest")
        fig.update_xaxes(title_text="SpO₂ (%)")
        fig.update_yaxes(title_text="days")
        charts.show(fig, key="h_spo2")
        low = (s < 95).mean()
        st.caption(f"{len(s):,} days with readings; median {s.median():.1f}%; {low:.0%} of days below 95%.")
