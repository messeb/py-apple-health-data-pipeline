import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui

ui.page_header("Body & gait", "Body composition and how you walk")
p = ui.period()
cur, prev, wide = ui.daily_frames(p)
pal = charts.palette()


def last(df, col):
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) else None


cols = st.columns(3) + st.columns(3)
with cols[0]:
    ui.kpi(
        "Body mass (latest)",
        last(cur, "body_mass_kg"),
        last(prev, "body_mass_kg"),
        ",.1f",
        "kg",
        neutral=True,
        help="Most recent weigh-in in the period vs the most recent one in the previous period",
    )
with cols[1]:
    ui.kpi("BMI (latest)", last(cur, "bmi"), last(prev, "bmi"), ",.1f", neutral=True)
for col, (label, c, fmt, unit, lower) in zip(
    cols[2:],
    [
        ("Walking speed", "walking_speed_kmh", ",.2f", "km/h", False),
        ("Step length", "step_length_cm", ",.1f", "cm", False),
        ("Walking asymmetry", "walking_asymmetry_pct", ",.1f", "%", True),
        ("Double-support time", "double_support_pct", ",.1f", "%", True),
    ],
    strict=False,
):
    with col:
        ui.kpi(label, ui.mean(cur, c), ui.mean(prev, c), fmt, unit, lower, spark=ui.weekly_spark(cur, c))

st.subheader("Body mass")
lo, hi = data.date_bounds()
wt = data.daily_summary(lo, hi)[["local_date", "body_mass_kg"]].dropna()
if wt.empty:
    ui.empty("No weigh-ins in the export. Body mass is only recorded when a scale or manual entry syncs.")
else:
    in_p = ui.in_range(wt, p)
    out_p = wt.drop(in_p.index)
    fig = go.Figure()
    fig.add_scatter(
        x=out_p["local_date"],
        y=out_p["body_mass_kg"],
        mode="markers",
        name="Other weigh-ins",
        marker=dict(size=6, color=pal.muted, opacity=0.6),
        hovertemplate="%{x|%d %b %Y}: %{y:.1f} kg<extra></extra>",
    )
    fig.add_scatter(
        x=in_p["local_date"],
        y=in_p["body_mass_kg"],
        mode="markers",
        name="In selected period",
        marker=dict(size=9, color=pal.series[0], line=dict(color=pal.surface, width=2)),
        hovertemplate="%{x|%d %b %Y}: %{y:.1f} kg<extra></extra>",
    )
    fig.add_vrect(x0=p.start, x1=p.end, fillcolor=pal.grid, opacity=0.35, line_width=0, layer="below")
    charts.style(fig, height=280, pal=pal).update_layout(hovermode="closest")
    fig.update_yaxes(title_text="kg")
    charts.show(fig, table=wt, key="b_mass")
    msg = f"All {len(wt)} weigh-ins since {wt['local_date'].min():%Y}; the shaded band is the selected period"
    if len(in_p) >= 2:
        change = in_p["body_mass_kg"].iloc[-1] - in_p["body_mass_kg"].iloc[0]
        msg += f" ({len(in_p)} weigh-ins, {change:+.1f} kg first to last)"
    st.caption(msg + ".")

st.subheader("Gait")
st.caption(
    "iPhone measures gait passively while you walk with it in a pocket. "
    "Higher speed and step length, lower asymmetry and double-support time are generally favourable."
)
GAIT = [
    ("Walking speed", "walking_speed_kmh", "km/h", ".2f"),
    ("Step length", "step_length_cm", "cm", ".1f"),
    ("Walking asymmetry", "walking_asymmetry_pct", "%", ".1f"),
    ("Double-support time", "double_support_pct", "%", ".1f"),
]
for row in (GAIT[:2], GAIT[2:]):
    cols = st.columns(2)
    for col, (title, c, unit, fmt) in zip(cols, row, strict=False):
        with col:
            st.markdown(f"**{title}**")
            if ui.in_range(wide, p)[c].notna().sum() == 0:
                ui.empty(f"No {title.lower()} data in this period.")
                continue
            charts.show(
                charts.trend(wide, c, unit, fmt=fmt, clip=(p.start, p.end), height=240, show_r7=False), key=f"b_{c}"
            )

steady = cur[["local_date", "walking_steadiness_pct"]].dropna()
if not steady.empty:
    st.markdown("**Walking steadiness**")
    fig = go.Figure(
        go.Scatter(
            x=steady["local_date"],
            y=steady["walking_steadiness_pct"],
            mode="lines+markers",
            line=dict(color=pal.series[0], width=2),
            marker=dict(size=7),
            hovertemplate="%{x|%d %b %Y}: %{y:.0f}%<extra></extra>",
        )
    )
    charts.style(fig, height=220, legend=False, pal=pal)
    fig.update_yaxes(title_text="%", range=[0, 102])
    charts.show(fig, key="b_steady")
    st.caption("Apple's fall-risk estimate from gait data, classified by Apple as OK, Low or Very Low.")
