"""Shared UI pieces: the global period filter, KPI tiles, empty states."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd
import streamlit as st

import data

PRESETS = {
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 12 months": 365,
    "Last 3 years": 3 * 365,
    "All time": None,
    "Custom": None,
}


@dataclass(frozen=True)
class Period:
    start: dt.date
    end: dt.date
    label: str

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def previous(self) -> Period:
        end = self.start - dt.timedelta(days=1)
        return Period(end - dt.timedelta(days=self.days - 1), end, "previous period")


def sidebar_period() -> Period:
    """The one filter row for every page: rendered in the sidebar by app.py."""
    lo, hi = data.date_bounds()
    preset = st.sidebar.selectbox("Period", list(PRESETS), index=2, key="period_preset")
    if preset == "Custom":
        picked = st.sidebar.date_input(
            "Date range", value=(hi - dt.timedelta(days=89), hi), min_value=lo, max_value=hi, key="period_custom"
        )
        if isinstance(picked, (list, tuple)):
            start, end = picked[0], (picked[1] if len(picked) > 1 else hi)
        else:
            start, end = picked, hi
    elif PRESETS[preset] is None:
        start, end = lo, hi
    else:
        start, end = max(lo, hi - dt.timedelta(days=PRESETS[preset] - 1)), hi
    period = Period(start, end, preset)
    st.session_state["period"] = period
    st.sidebar.caption(
        f"{start:%d %b %Y} – {end:%d %b %Y} · {period.days:,} days  \n"
        f"Deltas compare with the {period.days:,} days before."
    )
    return period


def period() -> Period:
    p = st.session_state.get("period")
    if p is None:
        lo, hi = data.date_bounds()
        p = Period(max(lo, hi - dt.timedelta(days=364)), hi, "Last 12 months")
    return p


def mean(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df or df[col].notna().sum() == 0:
        return None
    return float(df[col].mean())


def kpi(
    label: str,
    cur: float | None,
    prev: float | None,
    fmt: str = ",.0f",
    unit: str = "",
    lower_is_better: bool = False,
    help: str | None = None,
    spark: list[float] | None = None,
    neutral: bool = False,
) -> None:
    """A stat tile: value for the period, delta (absolute and relative) vs the previous period."""
    if cur is None:
        st.metric(label, "–", help=help or "No data in this period", border=True)
        return
    suffix = "" if not unit else unit if unit == "%" else f" {unit}"
    delta = None
    if prev is not None and prev != 0:
        change = cur - prev
        delta = f"{change:+{fmt}}{suffix} ({change / abs(prev):+.0%})"
    st.metric(
        label,
        f"{cur:{fmt}}{suffix}",
        delta=delta,
        delta_color="off" if neutral else "inverse" if lower_is_better else "normal",
        help=(help + " · " if help else "") + "Delta vs the previous period of equal length",
        border=True,
        chart_data=spark,
        chart_type="line",
    )


def empty(message: str) -> None:
    st.info(message, icon=":material/info:")


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    p = period()
    st.caption(f"{subtitle}  ·  {p.label}: {p.start:%d %b %Y} – {p.end:%d %b %Y}")


def in_range(df: pd.DataFrame, p: Period, col: str = "local_date") -> pd.DataFrame:
    return df[(df[col] >= pd.Timestamp(p.start)) & (df[col] <= pd.Timestamp(p.end))]


def daily_frames(p: Period) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(current, previous, wide) slices of gold_daily_summary from one query.

    `wide` adds a 28-day warm-up before the previous period for rolling means.
    """
    wide = data.daily_summary(p.previous.start - dt.timedelta(days=28), p.end)
    return in_range(wide, p), in_range(wide, p.previous), wide


def weekly_spark(df: pd.DataFrame, col: str) -> list[float] | None:
    if df.empty or df[col].notna().sum() < 3:
        return None
    s = df.set_index("local_date")[col].astype(float).resample("W").mean().dropna()
    return s.round(2).tolist() if len(s) >= 3 else None
