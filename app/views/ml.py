"""ML outputs from `ml/` (PyTorch). Each `data/ml/<stem>.parquet` (Databricks:
`ml_<stem>`) is one table. Known tables get dedicated views, grouped into tabs;
anything else lands in "Other tables" as a generic preview, so new model
outputs appear without code changes. Every view checks the columns it needs
and falls back to the generic preview otherwise.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui

ui.page_header("ML insights", "Outputs of the PyTorch models in ml/")
p = ui.period()
pal = charts.palette()
BASELINES = {"persistence", "seasonal_naive_7d", "mean_7d"}
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star", "hexagon"]

try:
    stems = data.list_ml_tables()
except Exception as exc:
    stems = []
    st.error(f"Could not list ML tables: {exc}")

if not stems:
    where = (
        f"`{data.data_dir() / 'ml'}`" if data.backend() == "local" else f"`{data.schema_name()}` (tables named `ml_*`)"
    )
    ui.empty(
        f"No ML outputs found in {where}. Run the training scripts in `ml/` "
        "(`.venv/bin/python ml/run_all.py`); they write their predictions as Parquet to `data/ml/`."
    )
    st.stop()


def load(stem: str, filter_period: bool = True) -> pd.DataFrame:
    df = data.ml_frame(stem)
    if filter_period and "local_date" in df.columns:
        df = ui.in_range(df.sort_values("local_date"), p)
    return df


def has(df: pd.DataFrame, *cols: str) -> bool:
    return set(cols) <= set(df.columns)


def pretty(name) -> str:
    return str(name).replace("_", " ")


def daily_context(dates: pd.Series) -> pd.DataFrame:
    """Key daily metrics for the given days, to explain model outputs."""
    if dates.empty:
        return pd.DataFrame()
    daily = data.daily_summary(dates.min().date(), dates.max().date())
    cols = [
        "local_date",
        "steps",
        "active_kcal",
        "exercise_min",
        "workout_min",
        "resting_hr",
        "hrv_sdnn_ms",
        "hr_mean",
        "sleep_asleep_min",
    ]
    return daily[[c for c in cols if c in daily.columns]]


def generic(df: pd.DataFrame, stem: str) -> None:
    st.markdown(f"**`{stem}`** · {len(df):,} rows · {len(df.columns)} columns")
    numeric = [c for c in df.select_dtypes(include=np.number).columns if c not in ("x", "y")]
    if "local_date" in df and numeric:
        col = st.selectbox("Column", numeric, key=f"ml_gen_{stem}")
        d = df.groupby("local_date")[col].mean().reset_index()
        fig = go.Figure(
            go.Scatter(
                x=d["local_date"],
                y=d[col],
                mode="lines",
                line=dict(color=pal.series[0], width=2),
                hovertemplate="%{x|%d %b %Y}: %{y:.3f}<extra></extra>",
            )
        )
        charts.style(fig, height=260, legend=False, pal=pal)
        fig.update_yaxes(title_text=pretty(col))
        charts.show(fig, key=f"ml_gen_chart_{stem}")
    st.dataframe(df.head(1000), hide_index=True)


# ---------------------------------------------------------------- unusual days


def view_anomalies() -> None:
    df = load("daily_anomalies")
    if df.empty or not has(df, "local_date", "anomaly_score"):
        return generic(df, "daily_anomalies") if not df.empty else ui.empty("No scored days in this period.")
    st.markdown(
        "An autoencoder learns what a typical day looks like (activity + vitals, relative to your recent "
        "baseline). Days it reconstructs poorly are **unusual**: illness, travel, unusually hard training "
        "or a sensor glitch."
    )
    flag = (
        df["is_anomaly"].fillna(False).astype(bool)
        if "is_anomaly" in df
        else df["anomaly_score"] > df["anomaly_score"].quantile(0.98)
    )
    cols = st.columns(3)
    cols[0].metric("Days scored", f"{len(df):,}", border=True)
    cols[1].metric("Flagged unusual", f"{int(flag.sum()):,}", border=True)
    cols[2].metric("Share flagged", f"{flag.mean():.1%}", border=True)

    fig = go.Figure()
    fig.add_scatter(
        x=df["local_date"],
        y=df["anomaly_score"],
        mode="lines",
        name="Autoencoder score",
        line=dict(color=pal.series[0], width=1.2),
        hovertemplate="%{x|%d %b %Y}: %{y:.3f}<extra>Autoencoder</extra>",
    )
    if "pca_anomaly_score" in df:
        fig.add_scatter(
            x=df["local_date"],
            y=df["pca_anomaly_score"],
            mode="lines",
            name="PCA score (reference)",
            line=dict(color=pal.series[1], width=1),
            opacity=0.8,
            hovertemplate="%{y:.3f}<extra>PCA</extra>",
        )
    f = df[flag]
    hover = f.get("top_features", None)
    fig.add_scatter(
        x=f["local_date"],
        y=f["anomaly_score"],
        mode="markers",
        name="Flagged day (autoencoder)",
        marker=dict(size=10, color=charts.STATUS["serious"], symbol="diamond", line=dict(color=pal.surface, width=2)),
        customdata=hover,
        hovertemplate="%{x|%d %b %Y}: %{y:.3f}"
        + ("<br>%{customdata}" if hover is not None else "")
        + "<extra>Flagged</extra>",
    )
    charts.style(fig, height=300, pal=pal)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="reconstruction error")
    charts.show(fig, key="ml_anom")
    if "pca_anomaly_score" in df:
        st.caption("PCA reconstruction error (same units) is shown as a linear baseline for the autoencoder.")

    st.markdown("**Most unusual days** and what drove the score")
    top = df.sort_values("anomaly_score", ascending=False).head(15)
    keep = [
        c
        for c in (
            "local_date",
            "anomaly_score",
            "top_features",
            "resting_hr",
            "resting_hr_dev",
            "hrv_sdnn_ms",
            "hrv_sdnn_ms_dev",
            "steps",
        )
        if c in top
    ]
    top = top[keep]
    ctx = daily_context(top["local_date"])
    if not ctx.empty:
        top = top.merge(ctx[[c for c in ctx if c not in top or c == "local_date"]], on="local_date", how="left")
    st.dataframe(
        top,
        hide_index=True,
        column_config={
            "local_date": st.column_config.DateColumn("Day", format="ddd DD MMM YYYY"),
            "anomaly_score": st.column_config.NumberColumn("Score", format="%.2f"),
            "top_features": st.column_config.TextColumn("Top features (deviation in SD)", width="large"),
            "resting_hr_dev": st.column_config.NumberColumn("RHR vs baseline", format="%+.1f"),
            "hrv_sdnn_ms_dev": st.column_config.NumberColumn("HRV vs baseline", format="%+.1f"),
            "steps": st.column_config.NumberColumn("Steps", format="%.0f"),
            "resting_hr": st.column_config.NumberColumn("Resting HR", format="%.0f"),
            "hrv_sdnn_ms": st.column_config.NumberColumn("HRV (ms)", format="%.0f"),
            "active_kcal": st.column_config.NumberColumn("Active kcal", format="%.0f"),
            "exercise_min": st.column_config.NumberColumn("Exercise min", format="%.0f"),
            "workout_min": st.column_config.NumberColumn("Workout min", format="%.0f"),
            "hr_mean": st.column_config.NumberColumn("Mean HR", format="%.0f"),
            "sleep_asleep_min": st.column_config.NumberColumn("Asleep min", format="%.0f"),
        },
    )

    if "anomaly_episodes" in stems:
        ep = data.ml_frame("anomaly_episodes")
        if has(ep, "start", "end"):
            ep = ep.assign(start=pd.to_datetime(ep["start"]), end=pd.to_datetime(ep["end"]))
            ep = ep[(ep["end"] >= pd.Timestamp(p.start)) & (ep["start"] <= pd.Timestamp(p.end))]
            st.markdown(f"**Episodes**: consecutive unusual days merged ({len(ep)} in this period)")
            sort_col = "max_score" if "max_score" in ep else "start"
            st.dataframe(
                ep.sort_values(sort_col, ascending=False),
                hide_index=True,
                column_config={
                    "start": st.column_config.DateColumn("From", format="DD MMM YYYY"),
                    "end": st.column_config.DateColumn("To", format="DD MMM YYYY"),
                    "n_days": "Days",
                    "max_score": st.column_config.NumberColumn("Peak score", format="%.2f"),
                    "mean_rhr_dev": st.column_config.NumberColumn("Mean RHR vs baseline (bpm)", format="%+.1f"),
                    "mean_hrv_dev": st.column_config.NumberColumn("Mean HRV vs baseline (ms)", format="%+.1f"),
                },
            )

    if "illness_score" in df:
        illness_like(df)


def illness_like(df: pd.DataFrame) -> None:
    """Days with resting HR above and HRV below the personal baseline (a classic strain/illness pattern)."""
    st.markdown("**Illness-like days**: resting HR up *and* HRV down vs your 60-day baseline")
    st.caption(
        "Ranked by `illness_score` = z(resting HR deviation) − z(HRV deviation). The autoencoder score above does "
        "not rank these days better than chance (AUC 0.50 in ml/reports/RESULTS.md), so this view uses its own score. "
        "A pattern, not a diagnosis: hard training, alcohol, heat or poor sleep look the same."
    )
    d = df.dropna(subset=["illness_score"])
    if d.empty:
        return ui.empty("No days with an illness score in this period.")
    fig = go.Figure()
    fig.add_scatter(
        x=d["local_date"],
        y=d["illness_score"],
        mode="lines",
        name="Illness score",
        line=dict(color=pal.series[0], width=1.2),
        hovertemplate="%{x|%d %b %Y}: %{y:+.2f}<extra></extra>",
    )
    top = d.sort_values("illness_score", ascending=False).head(10)
    fig.add_scatter(
        x=top["local_date"],
        y=top["illness_score"],
        mode="markers",
        name="Top 10 in period",
        marker=dict(size=10, color=charts.STATUS["serious"], symbol="diamond", line=dict(color=pal.surface, width=2)),
        hovertemplate="%{x|%d %b %Y}: %{y:+.2f}<extra>Top 10</extra>",
    )
    fig.add_hline(y=0, line=dict(color=pal.axis, width=1))
    charts.style(fig, height=260, pal=pal).update_layout(hovermode="closest")
    fig.update_yaxes(title_text="illness score (z)")
    charts.show(fig, key="ml_illness")
    keep = [
        c
        for c in (
            "local_date",
            "illness_score",
            "resting_hr",
            "resting_hr_dev",
            "hrv_sdnn_ms",
            "hrv_sdnn_ms_dev",
            "steps",
            "anomaly_score",
        )
        if c in top
    ]
    st.dataframe(
        top[keep],
        hide_index=True,
        column_config={
            "local_date": st.column_config.DateColumn("Day", format="ddd DD MMM YYYY"),
            "illness_score": st.column_config.NumberColumn("Illness score", format="%+.2f"),
            "resting_hr": st.column_config.NumberColumn("Resting HR", format="%.0f"),
            "resting_hr_dev": st.column_config.NumberColumn("RHR vs baseline", format="%+.1f"),
            "hrv_sdnn_ms": st.column_config.NumberColumn("HRV (ms)", format="%.0f"),
            "hrv_sdnn_ms_dev": st.column_config.NumberColumn("HRV vs baseline", format="%+.1f"),
            "steps": st.column_config.NumberColumn("Steps", format="%.0f"),
            "anomaly_score": st.column_config.NumberColumn("Autoencoder score", format="%.2f"),
        },
    )


# ---------------------------------------------------------------- forecasts


def view_forecasts() -> None:
    metrics = data.ml_frame("forecast_metrics") if "forecast_metrics" in stems else pd.DataFrame()
    targets = [s.removeprefix("forecast_") for s in stems if s.startswith("forecast_") and s != "forecast_metrics"]
    if not targets:
        return generic(metrics, "forecast_metrics")
    labels = {"resting_hr": "Resting heart rate (bpm)", "hrv_sdnn_ms": "HRV SDNN (ms)"}
    target = (
        st.segmented_control(
            "Target", targets, default=targets[0], format_func=lambda t: labels.get(t, pretty(t)), key="ml_fc_target"
        )
        or targets[0]
    )
    df = load(f"forecast_{target}")
    if not has(df, "local_date", "actual", "predicted"):
        return generic(df, f"forecast_{target}")
    if "model" not in df:
        df = df.assign(model="model")
    st.markdown(
        "Next-day forecasts on held-out days. Simple baselines (*persistence* = tomorrow equals today, "
        "*mean 7d*) are the bar a model has to clear."
    )

    test = pd.DataFrame()
    if has(metrics, "target", "model", "mae"):
        test = metrics[metrics["target"] == target]
        if "split" in test:
            test = test[test["split"] == "test"] if (test["split"] == "test").any() else test
        test = test.sort_values("mae")
    all_models = list(test["model"]) if not test.empty else sorted(df["model"].dropna().unique())
    all_models = [m for m in all_models if m in set(df["model"])]
    default = all_models[:1] + [m for m in ("persistence",) if m in all_models and m not in all_models[:1]]
    models = st.multiselect("Models", all_models, default=default, key=f"ml_fc_models_{target}") or default

    if not test.empty:
        best = test.iloc[0]
        cols = st.columns(3)
        cols[0].metric("Best model (test MAE)", pretty(best["model"]), border=True)
        cols[1].metric("Test MAE", f"{best['mae']:.2f}", border=True, help="Mean absolute error on the test split")
        if "mae_skill_vs_persistence" in best and pd.notna(best["mae_skill_vs_persistence"]):
            cols[2].metric(
                "Skill vs persistence",
                f"{best['mae_skill_vs_persistence']:+.0%}",
                border=True,
                help="1 - MAE(model) / MAE(persistence); higher is better",
            )

    if df.empty:
        ui.empty("No forecast rows in the selected period; forecasts cover the validation/test years only.")
    else:
        # Baselines share the secondary ink; learned models take categorical slots in leaderboard order.
        learned = [m for m in all_models if m not in BASELINES]
        cmap = {
            **{m: pal.series[i % len(pal.series)] for i, m in enumerate(learned)},
            **{m: pal.ink2 for m in all_models if m in BASELINES},
        }
        fig = go.Figure()
        actual = df.drop_duplicates("local_date")[["local_date", "actual"]].dropna()
        fig.add_scatter(
            x=actual["local_date"],
            y=actual["actual"],
            mode="markers",
            name="Actual",
            marker=dict(size=5, color=pal.muted),
            hovertemplate="%{y:.1f}<extra>Actual</extra>",
        )
        for m in models:
            d = df[df["model"] == m].dropna(subset=["predicted"])
            fig.add_scatter(
                x=d["local_date"],
                y=d["predicted"],
                mode="lines",
                name=pretty(m),
                line=dict(color=cmap[m], width=1.2 if m in BASELINES else 2),
                hovertemplate=f"%{{y:.1f}}<extra>{pretty(m)}</extra>",
            )
        charts.style(fig, height=320, pal=pal)
        fig.update_yaxes(title_text=labels.get(target, pretty(target)))
        charts.show(fig, key=f"ml_fc_{target}")
        err = (
            df.dropna(subset=["actual", "predicted"])
            .assign(abs_err=lambda d: (d["actual"] - d["predicted"]).abs())
            .groupby("model")["abs_err"]
            .mean()
        )
        if not err.empty:
            st.caption(
                "MAE within the selected period: " + " · ".join(f"{pretty(m)} {err[m]:.2f}" for m in models if m in err)
            )

    if not test.empty:
        st.markdown("**Leaderboard** (test split)")
        bars = test.sort_values("mae", ascending=False)
        colors = [pal.neutral if m in BASELINES else pal.series[0] for m in bars["model"]]
        fig = go.Figure(
            go.Bar(
                y=bars["model"].map(pretty),
                x=bars["mae"],
                orientation="h",
                marker=dict(color=colors, cornerradius=4),
                hovertemplate="%{y}: MAE %{x:.2f}<extra></extra>",
            )
        )
        charts.style(fig, height=26 * len(bars) + 70, legend=False, pal=pal).update_layout(hovermode="closest")
        fig.update_xaxes(title_text="MAE (lower is better)", showgrid=True)
        fig.update_yaxes(showgrid=False)
        charts.show(fig, table=test, key=f"ml_fc_board_{target}")
        st.caption("Gray bars are naive baselines.")


# ---------------------------------------------------------------- clusters


def embedding(df: pd.DataFrame, group_col: str, label_of, key: str, highlight: str | None, hover_col: str | None):
    groups = sorted(df[group_col].dropna().unique())
    fig = go.Figure()
    cmap = {g: pal.series[i % len(pal.series)] for i, g in enumerate(groups)}
    shown_other = False
    for i, g in enumerate(groups):
        d = df[df[group_col] == g]
        on = highlight in (None, g)
        legend = on or not shown_other
        shown_other = shown_other or not on
        fig.add_scattergl(
            x=d["x"],
            y=d["y"],
            mode="markers",
            name=label_of(g) if on else "Other clusters",
            showlegend=legend,
            legendgroup="on" if on else "other",
            marker=dict(
                size=7,
                color=cmap[g] if on else pal.neutral,
                opacity=0.8 if on else 0.35,
                symbol=SYMBOLS[i % len(SYMBOLS)] if on else "circle",
                line=dict(color=pal.surface, width=1),
            ),
            customdata=d[hover_col] if hover_col else None,
            hovertemplate=f"{label_of(g)}" + ("<br>%{customdata}" if hover_col else "") + "<extra></extra>",
        )
    charts.style(fig, height=400, pal=pal).update_layout(hovermode="closest")
    fig.update_xaxes(showticklabels=False, showgrid=False, title_text="embedding 1")
    fig.update_yaxes(showticklabels=False, showgrid=False, title_text="embedding 2")
    charts.show(fig, key=key)


def view_day_clusters() -> None:
    df = load("day_clusters")
    if df.empty or "cluster" not in df:
        return generic(df, "day_clusters") if not df.empty else ui.empty("No clustered days in this period.")
    names = (
        df.drop_duplicates("cluster").set_index("cluster")["cluster_label"].to_dict() if "cluster_label" in df else {}
    )

    def label_of(c):
        return f"{c} · {names[c]}" if c in names else f"Cluster {c}"

    st.markdown(
        "Days grouped by *behaviour* (steps, distances, stairs, energy, exercise, stand and workout time; "
        "k-means on standardized features, PCA projection for display). Heart metrics are not clustering "
        "inputs; they are reported per cluster."
    )
    left, right = st.columns([3, 2])
    with left:
        if has(df, "x", "y"):
            pick = st.selectbox(
                "Highlight",
                ["All", *sorted(df["cluster"].unique())],
                key="ml_dc_hl",
                format_func=lambda c: c if c == "All" else label_of(c),
            )
            embedding(
                df.assign(day=df["local_date"].dt.strftime("%a %d %b %Y")),
                "cluster",
                label_of,
                "ml_dc_xy",
                None if pick == "All" else pick,
                "day",
            )
    with right:
        st.markdown("**Cluster profiles** (mean per day, all time)")
        if "day_cluster_profiles" in stems:
            prof = data.ml_frame("day_cluster_profiles")
            cols = [
                c
                for c in (
                    "cluster",
                    "cluster_label",
                    "n_days",
                    "steps",
                    "active_kcal",
                    "exercise_min",
                    "workout_min",
                    "cycling_km",
                    "resting_hr",
                    "hrv_sdnn_ms",
                    "is_weekend",
                )
                if c in prof
            ]
            st.dataframe(
                prof[cols],
                hide_index=True,
                column_config={
                    "cluster_label": "Label",
                    "n_days": "Days",
                    "steps": st.column_config.NumberColumn("Steps", format="%.0f"),
                    "active_kcal": st.column_config.NumberColumn("Active kcal", format="%.0f"),
                    "exercise_min": st.column_config.NumberColumn("Exercise min", format="%.0f"),
                    "workout_min": st.column_config.NumberColumn("Workout min", format="%.0f"),
                    "cycling_km": st.column_config.NumberColumn("Cycling km", format="%.1f"),
                    "resting_hr": st.column_config.NumberColumn("Resting HR", format="%.1f"),
                    "hrv_sdnn_ms": st.column_config.NumberColumn("HRV", format="%.0f"),
                    "is_weekend": st.column_config.NumberColumn("Weekend share", format="percent"),
                },
            )
        else:
            ctx = daily_context(df["local_date"])
            prof = df[["local_date", "cluster"]].merge(ctx, on="local_date").drop(columns="local_date")
            st.dataframe(prof.groupby("cluster").mean(numeric_only=True).round(1).reset_index(), hide_index=True)

    st.markdown("**Day types over time** (share of days per month)")
    m = df.assign(month=df["local_date"].dt.to_period("M").dt.to_timestamp())
    share = m.pivot_table(index="month", columns="cluster", values="local_date", aggfunc="count").fillna(0)
    share = share.div(share.sum(axis=1), axis=0) * 100
    groups = sorted(df["cluster"].unique())
    fig = go.Figure()
    for i, g in enumerate(groups):
        if g in share:
            fig.add_bar(
                x=share.index,
                y=share[g],
                name=label_of(g),
                marker=dict(color=pal.series[i % len(pal.series)], line=dict(color=pal.surface, width=1)),
                hovertemplate=f"{label_of(g)}: %{{y:.0f}}%<extra></extra>",
            )
    charts.style(fig, height=320, pal=pal).update_layout(barmode="stack", margin=dict(t=64))
    fig.update_yaxes(title_text="% of days", range=[0, 100])
    charts.show(fig, table=share.round(0).rename(columns=label_of).reset_index(), key="ml_dc_time")


def view_similar_days() -> None:
    df = data.ml_frame("similar_days")
    if not has(df, "local_date", "neighbor_date"):
        return generic(df, "similar_days")
    st.markdown(
        "Nearest neighbours in the anomaly autoencoder's embedding: other days (excluding the days right "
        "around it) that *looked like* the chosen day."
    )
    days = sorted(ui.in_range(df, p)["local_date"].unique(), reverse=True)
    if not days:
        return ui.empty("No days with neighbours in this period.")
    day = st.selectbox("Day", days, format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b %Y"), key="ml_sim_day")
    n = df[df["local_date"] == day].sort_values("neighbor_rank" if "neighbor_rank" in df else "neighbor_date")
    dates = pd.concat([pd.Series([pd.Timestamp(day)]), pd.to_datetime(n["neighbor_date"])])
    lo, hi = data.date_bounds()
    ctx = data.daily_summary(lo, hi)
    ctx = ctx[ctx["local_date"].isin(dates)]
    cols = ["local_date", "steps", "active_kcal", "exercise_min", "workout_min", "resting_hr", "hrv_sdnn_ms"]
    out = pd.DataFrame(
        {
            "local_date": dates.values,
            "role": ["Selected day"] + [f"Neighbour {i + 1}" for i in range(len(n))],
            "distance": [0.0, *list(n["distance"] if "distance" in n else [np.nan] * len(n))],
        }
    ).merge(ctx[[c for c in cols if c in ctx]], on="local_date", how="left")
    st.dataframe(
        out,
        hide_index=True,
        column_config={
            "local_date": st.column_config.DateColumn("Day", format="ddd DD MMM YYYY"),
            "role": "",
            "distance": st.column_config.NumberColumn("Distance", format="%.2f"),
            "steps": st.column_config.NumberColumn("Steps", format="%.0f"),
            "active_kcal": st.column_config.NumberColumn("Active kcal", format="%.0f"),
            "exercise_min": st.column_config.NumberColumn("Exercise min", format="%.0f"),
            "workout_min": st.column_config.NumberColumn("Workout min", format="%.0f"),
            "resting_hr": st.column_config.NumberColumn("Resting HR", format="%.0f"),
            "hrv_sdnn_ms": st.column_config.NumberColumn("HRV", format="%.0f"),
        },
    )


def view_workout_clusters() -> None:
    df = load("workout_clusters")
    if df.empty or not has(df, "cluster"):
        return generic(df, "workout_clusters") if not df.empty else ui.empty("No clustered workouts in this period.")
    st.markdown(
        "Workouts grouped by duration, energy rate, heart rate, speed, start hour and indoor/outdoor, "
        "independent of the activity label."
    )
    w = data.workouts(p.start, p.end)
    if "workout_id" in df:
        df = df.merge(
            w[["workout_id", "duration_min", "distance_km", "hr_avg", "active_kcal"]], on="workout_id", how="left"
        )
    summary = (
        df.groupby("cluster")
        .agg(
            workouts=("cluster", "size"),
            main_activities=("activity", lambda s: ", ".join(f"{a} ({c})" for a, c in s.value_counts().head(3).items()))
            if "activity" in df
            else ("cluster", "size"),
            **{
                k: (v, "median")
                for k, v in (
                    ("median_min", "duration_min"),
                    ("median_km", "distance_km"),
                    ("median_hr", "hr_avg"),
                    ("median_kcal", "active_kcal"),
                )
                if v in df
            },
        )
        .reset_index()
    )

    def label_of(c):
        return f"Cluster {c}"

    left, right = st.columns([3, 2])
    with left:
        if has(df, "x", "y"):
            groups = sorted(df["cluster"].unique())
            pick = st.selectbox("Highlight", groups, index=0, key="ml_wc_hl", format_func=label_of)
            hover = (
                df["activity"].astype(str) + " · " + df["local_date"].dt.strftime("%d %b %Y")
                if "activity" in df
                else None
            )
            embedding(
                df.assign(h=hover) if hover is not None else df,
                "cluster",
                label_of,
                "ml_wc_xy",
                pick,
                "h" if hover is not None else None,
            )
            st.caption(
                "One cluster is highlighted at a time; eight hues on one scatter would not stay distinguishable."
            )
    with right:
        st.markdown("**What each cluster contains**")
        st.dataframe(
            summary,
            hide_index=True,
            column_config={
                "main_activities": st.column_config.TextColumn("Main activities", width="large"),
                "median_min": st.column_config.NumberColumn("Median min", format="%.0f"),
                "median_km": st.column_config.NumberColumn("Median km", format="%.1f"),
                "median_hr": st.column_config.NumberColumn("Median HR", format="%.0f"),
                "median_kcal": st.column_config.NumberColumn("Median kcal", format="%.0f"),
            },
        )


FLAG_STYLES = {  # flag_reason -> (legend label, status color, marker)
    "shape": ("Unusual shape", "serious", "diamond"),
    "rr_interval": ("Unusual R-R interval", "warning", "triangle-up"),
    "shape+rr_interval": ("Shape and R-R interval", "critical", "star"),
}


def view_ecg() -> None:
    rec = data.ml_frame("ecg_recordings") if "ecg_recordings" in stems else pd.DataFrame()
    beats = data.ml_frame("ecg_beat_anomalies") if "ecg_beat_anomalies" in stems else pd.DataFrame()
    st.markdown(
        "A 1-D convolutional autoencoder learns the shape of a normal beat. Scores are reconstruction error "
        "relative to a typical training beat (1 = typical). A beat is also flagged when the R-R interval "
        "before it is far from the recording's median. All recordings, regardless of the period filter."
    )
    if has(rec, "ecg_id", "recorded_local"):
        rec = rec.sort_values("recorded_local")
        cols = st.columns(3)
        if "hr_bpm" in rec:
            cols[0].metric("Median ECG heart rate", f"{rec['hr_bpm'].median():.0f} bpm", border=True)
        if "n_anomalous_beats" in rec:
            cols[1].metric(
                "Recordings with flagged beats",
                f"{int((rec['n_anomalous_beats'] > 0).sum())} / {len(rec)}",
                border=True,
            )
        if "rmssd_ms" in rec:
            cols[2].metric("Median RMSSD", f"{rec['rmssd_ms'].median():.0f} ms", border=True)
        x = rec["recorded_local"].dt.strftime("%Y-%m-%d %H:%M")
        parts = (
            {"n_shape_flags": "Shape flags", "n_rr_flags": "R-R interval flags"}
            if has(rec, "n_shape_flags", "n_rr_flags")
            else {"n_anomalous_beats": "Flagged beats"}
            if "n_anomalous_beats" in rec
            else {}
        )
        if parts:
            st.markdown("**Flagged beats per recording**")
            fig = go.Figure()
            for i, (c, name) in enumerate(parts.items()):
                fig.add_bar(
                    x=x,
                    y=rec[c],
                    name=name,
                    marker=dict(color=pal.series[i], line=dict(color=pal.surface, width=1)),
                    hovertemplate=f"%{{x}}: %{{y}} {name.lower()}<extra></extra>",
                )
            charts.style(fig, height=280, legend=len(parts) > 1, pal=pal).update_layout(
                barmode="stack", hovermode="closest"
            )
            fig.update_xaxes(type="category", tickangle=-45)
            fig.update_yaxes(title_text="beats")
            charts.show(fig, table=rec, key="ml_ecg_rec")
            st.caption("A beat can carry both flags; it is then counted in both parts of the bar.")
    if not has(beats, "ecg_id", "anomaly_score"):
        if not beats.empty:
            generic(beats, "ecg_beat_anomalies")
        return

    st.markdown("**Beat scores in one recording**")
    ids = list(beats.sort_values("anomaly_score", ascending=False)["ecg_id"].drop_duplicates())
    when = (
        beats.drop_duplicates("ecg_id").set_index("ecg_id")["recorded_local"].dt.strftime("%d %b %Y, %H:%M").to_dict()
        if "recorded_local" in beats
        else {}
    )
    labels = {
        i: f"{when.get(i, i)} · max score {beats.loc[beats['ecg_id'] == i, 'anomaly_score'].max():,.1f}" for i in ids
    }
    by_label = {v: k for k, v in labels.items()}
    ecg_id = by_label[st.selectbox("Recording (highest-scoring first)", list(by_label), key="ml_ecg_id")]
    b = beats[beats["ecg_id"] == ecg_id].sort_values("beat_index" if "beat_index" in beats else "anomaly_score")
    b = b.assign(
        t=b["r_peak_s"] if "r_peak_s" in b else range(len(b)),
        reason=b["flag_reason"]
        if "flag_reason" in b
        else b.get("is_anomaly", pd.Series(False, index=b.index)).map({True: "shape"}),
        rr=b.get("rr_ms", np.nan),
    )
    hover = "%{x:.1f} s · score %{y:,.2f}<br>R-R %{customdata:.0f} ms"
    fig = go.Figure()
    normal = b[b["reason"].isna()]
    fig.add_scatter(
        x=normal["t"],
        y=normal["anomaly_score"],
        mode="markers",
        name="Not flagged",
        marker=dict(size=7, color=pal.muted, opacity=0.7),
        customdata=normal["rr"],
        hovertemplate=hover + "<extra>Not flagged</extra>",
    )
    for reason, (label, status, symbol) in FLAG_STYLES.items():
        d = b[b["reason"] == reason]
        if d.empty:
            continue
        fig.add_scatter(
            x=d["t"],
            y=d["anomaly_score"],
            mode="markers",
            name=label,
            customdata=d["rr"],
            marker=dict(size=12, color=charts.STATUS[status], symbol=symbol, line=dict(color=pal.surface, width=2)),
            hovertemplate=hover + f"<extra>{label}</extra>",
        )
    fig.add_hline(
        y=1,
        line=dict(color=pal.ink2, width=1),
        annotation=dict(text="typical beat", font=dict(color=pal.ink2, size=11)),
        annotation_position="bottom right",
    )
    charts.style(fig, height=300, pal=pal).update_layout(hovermode="closest")
    fig.update_xaxes(title_text="seconds into recording" if "r_peak_s" in b else "beat", showgrid=True)
    fig.update_yaxes(title_text="relative reconstruction error (log)", type="log", dtick=1)
    keep = [
        c for c in ("beat_index", "r_peak_s", "rr_ms", "anomaly_score", "reconstruction_mse", "flag_reason") if c in b
    ]
    charts.show(fig, table=b[keep], key="ml_ecg_beats")
    st.caption(
        "Log scale: scores range from below 1 to over a thousand. Open this recording on the ECG page "
        "to see the waveform."
    )


# ---------------------------------------------------------------- layout

SECTIONS = [
    ("Unusual days", {"daily_anomalies", "anomaly_episodes"}, view_anomalies),
    ("Forecasts", {s for s in stems if s.startswith("forecast_")}, view_forecasts),
    ("Day types", {"day_clusters", "day_cluster_profiles"}, view_day_clusters),
    ("Similar days", {"similar_days"}, view_similar_days),
    ("Workout clusters", {"workout_clusters"}, view_workout_clusters),
    ("ECG beats", {"ecg_recordings", "ecg_beat_anomalies"}, view_ecg),
]
present = [(title, tables, fn) for title, tables, fn in SECTIONS if tables & set(stems)]
known = set().union(*(t for _, t, _ in SECTIONS))
others = [s for s in stems if s not in known]

tabs = st.tabs([t for t, _, _ in present] + (["Other tables"] if others else []))
for tab, (title, tables, fn) in zip(tabs, present, strict=False):
    with tab:
        try:
            fn()
        except Exception as exc:  # one malformed output must not take down the page
            st.error(f"Could not render {title.lower()}: {exc}")
        st.caption("Source tables: " + ", ".join(f"`{data.ml_table(s)}`" for s in sorted(tables & set(stems))))
if others:
    with tabs[-1]:
        for stem in others:
            try:
                generic(load(stem, filter_period=False), stem)
            except Exception as exc:
                st.error(f"Could not read `{stem}`: {exc}")
