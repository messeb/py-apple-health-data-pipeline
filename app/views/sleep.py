import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui

ui.page_header("Sleep", "What the export does (and does not) say about sleep")
p = ui.period()
pal = charts.palette()

lo, hi = data.date_bounds()
allnights = data.sleep(lo, hi)
nights = ui.in_range(allnights, p, "night_date")

# ---------------------------------------------------------------- honesty first
n_asleep = int(allnights["asleep_min"].notna().sum())
n_stages = int(allnights["has_stages"].fillna(False).sum())
n_inbed = int((allnights["in_bed_min"].notna() & allnights["asleep_min"].isna()).sum())
med_asleep = allnights["asleep_min"].median()
coverage = f"{len(allnights):,} nights with any sleep record, {n_asleep:,} with *asleep* time and {n_stages:,} with sleep stages."
sparse = len(allnights) == 0 or n_asleep < 0.6 * len(allnights) or (pd.notna(med_asleep) and med_asleep < 300)
if sparse:
    st.warning(
        f"**Sleep data is sparse.** Across all time there are {coverage} {n_inbed:,} nights have only an "
        f"*in bed* window (iPhone Bedtime), which measures time in bed, not sleep. "
        + (
            f"Recorded asleep periods have a median of {med_asleep:.0f} min, so many are naps or partial nights. "
            if pd.notna(med_asleep) and med_asleep < 300
            else ""
        )
        + "Treat everything on this page as indicative.",
        icon=":material/warning:",
    )
else:
    st.caption(f"Across all time: {coverage}")

st.subheader("Coverage by year")
cov = (
    allnights.assign(year=allnights["night_date"].dt.year)
    .groupby("year")
    .agg(
        any_record=("night_date", "count"),
        in_bed=("in_bed_min", "count"),
        asleep=("asleep_min", "count"),
        stages=("has_stages", lambda s: int(s.fillna(False).sum())),
    )
    .reset_index()
)
names = {"in_bed": "In-bed window", "asleep": "Asleep time", "stages": "With stages"}
cmap = charts.color_map(list(names.values()), pal)
fig = go.Figure()
for c, name in names.items():
    fig.add_bar(
        x=cov["year"].astype(str),
        y=cov[c],
        name=name,
        marker=dict(color=cmap[name], cornerradius=4),
        hovertemplate=f"{name}: %{{y}} nights<extra></extra>",
    )
charts.style(fig, height=280, pal=pal).update_layout(barmode="group", bargroupgap=0.1)
fig.update_yaxes(title_text="nights")
charts.show(fig, table=cov.rename(columns=names), key="s_cov")

if nights.empty:
    ui.empty("No sleep records in the selected period.")
    st.stop()

# ---------------------------------------------------------------- in-period views
st.subheader("In the selected period")
cols = st.columns(4)
cols[0].metric("Nights with a record", f"{len(nights):,}", border=True)
cols[1].metric(
    "Median time in bed",
    f"{nights['in_bed_min'].median() / 60:.1f} h" if nights["in_bed_min"].notna().any() else "–",
    border=True,
)
cols[2].metric(
    "Median recorded asleep",
    f"{nights['asleep_min'].median() / 60:.1f} h" if nights["asleep_min"].notna().any() else "–",
    border=True,
)
cols[3].metric("Nights with stages", f"{int(nights['has_stages'].fillna(False).sum()):,}", border=True)

left, right = st.columns(2)
with left:
    st.markdown("**Duration per night**")
    fig = go.Figure()
    series = {"in_bed_min": "In bed", "asleep_min": "Asleep"}
    cmap2 = charts.color_map(list(series.values()), pal)
    for c, name in series.items():
        d = nights[["night_date", c]].dropna()
        if d.empty:
            continue
        fig.add_scatter(
            x=d["night_date"],
            y=d[c] / 60,
            mode="markers",
            name=name,
            marker=dict(size=7, color=cmap2[name], opacity=0.8, line=dict(color=pal.surface, width=1)),
            hovertemplate=f"{name}: %{{y:.1f}} h<extra></extra>",
        )
    charts.style(fig, height=300, pal=pal)
    fig.update_yaxes(title_text="hours", rangemode="tozero")
    charts.show(fig, key="s_dur")
with right:
    st.markdown("**When the night starts**")
    b = nights["bedtime_hours_after_18"].dropna()
    if b.empty:
        ui.empty("No bedtimes recorded.")
    else:
        ticks = list(range(0, 25, 3))
        fig = go.Figure(
            go.Histogram(
                x=b,
                xbins=dict(start=0, end=24, size=0.5),
                marker=dict(color=pal.series[0], line=dict(color=pal.surface, width=2)),
                hovertemplate="%{y} nights<extra></extra>",
            )
        )
        charts.style(fig, height=300, legend=False, pal=pal).update_layout(hovermode="closest")
        fig.update_xaxes(
            tickvals=ticks,
            ticktext=[f"{(18 + t) % 24:02d}:00" for t in ticks],
            range=[0, 24],
            title_text="start of first sleep/in-bed segment",
        )
        fig.update_yaxes(title_text="nights")
        charts.show(fig, key="s_bed")
        med = b.median()
        hh, mm = divmod(round(((18 + med) % 24) * 60), 60)
        st.caption(f"Median start {hh:02d}:{mm:02d}. Daytime starts are naps.")

staged = nights[nights["has_stages"].fillna(False)]
if not staged.empty:
    st.markdown("**Sleep stages (nights tracked by the Watch)**")
    stages = {"deep_min": "Deep", "core_min": "Core", "rem_min": "REM", "awake_min": "Awake"}
    smap = charts.color_map(list(stages.values()), pal)
    fig = go.Figure()
    for c, name in stages.items():
        fig.add_bar(
            x=staged["night_date"].dt.strftime("%d %b %Y"),
            y=staged[c].fillna(0) / 60,
            name=name,
            marker=dict(color=smap[name], line=dict(color=pal.surface, width=1)),
            hovertemplate=f"{name}: %{{y:.1f}} h<extra></extra>",
        )
    charts.style(fig, height=280, pal=pal).update_layout(barmode="stack")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="hours")
    charts.show(fig, table=staged[["night_date", *stages]].rename(columns=stages), key="s_stages")
