"""Chart theme and Plotly helpers.

Colors come from the validated reference palette (dataviz skill,
references/palette.md): categorical slots in fixed order, a one-hue blue
sequential ramp, a blue<->red diverging pair with a gray midpoint, and status
colors reserved for good/warning/serious/critical. Light and dark are selected
steps, chosen from the active Streamlit theme.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

SEQ_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


@dataclass(frozen=True)
class Palette:
    mode: str
    surface: str
    page: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    neutral: str  # diverging midpoint / "everything else" gray
    red: str


LIGHT = Palette(
    mode="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    neutral="#c3c2b7",
    red="#e34948",
)
DARK = Palette(
    mode="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    ink="#ffffff",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    neutral="#55544f",
    red="#e66767",
)


def palette() -> Palette:
    try:
        kind = st.context.theme.type
    except Exception:  # outside a script run (tests, bare mode)
        kind = None
    return DARK if kind == "dark" else LIGHT


def seq_scale(pal: Palette | None = None) -> list[list]:
    """Sequential blue; low values recede toward the surface in both modes."""
    pal = pal or palette()
    ramp = SEQ_RAMP if pal.mode == "light" else SEQ_RAMP[::-1]
    ramp = ramp[:-2] if pal.mode == "light" else ramp[2:]
    n = len(ramp) - 1
    return [[i / n, c] for i, c in enumerate(ramp)]


def ordinal(n: int, pal: Palette | None = None) -> list[str]:
    """n ordered steps of the blue ramp that all clear 2:1 against the surface."""
    pal = pal or palette()
    usable = SEQ_RAMP[3:] if pal.mode == "light" else SEQ_RAMP[:11]
    if n == 1:
        return [usable[len(usable) // 2]]
    idx = [round(i * (len(usable) - 1) / (n - 1)) for i in range(n)]
    steps = [usable[i] for i in idx]
    return steps if pal.mode == "light" else steps[::-1]


def color_map(keys: list[str], pal: Palette | None = None) -> dict[str, str]:
    """Fixed slot per entity (in the given stable order); 'Other' is gray."""
    pal = pal or palette()
    out, slot = {}, 0
    for k in keys:
        if k == "Other":
            out[k] = pal.neutral
        else:
            out[k] = pal.series[slot % len(pal.series)]
            slot += 1
    return out


def style(fig: go.Figure, height: int = 320, legend: bool = True, pal: Palette | None = None) -> go.Figure:
    pal = pal or palette()
    axis = dict(
        gridcolor=pal.grid,
        gridwidth=1,
        zeroline=False,
        linecolor=pal.axis,
        linewidth=1,
        tickfont=dict(color=pal.muted, size=11),
        title_font=dict(color=pal.ink2, size=12),
        ticks="",
        showline=False,
        automargin=True,
    )
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=pal.ink2, size=12),
        margin=dict(l=8, r=8, t=36 if legend else 12, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=pal.surface, bordercolor=pal.axis, font=dict(color=pal.ink, family=FONT)),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="left",
            x=0,
            font=dict(color=pal.ink2),
            bgcolor="rgba(0,0,0,0)",
            title_text="",
            traceorder="normal",
        ),
        colorway=list(pal.series),
        bargap=0.2,
    )
    fig.update_xaxes(**axis, showgrid=False)
    fig.update_yaxes(**axis, showgrid=True)
    return fig


def show(fig: go.Figure, table: pd.DataFrame | None = None, key: str | None = None) -> None:
    """Render a figure; optionally a table-view twin behind an expander."""
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False, "responsive": True})
    if table is not None and not table.empty:
        table = table.copy()
        floats = table.select_dtypes("float").columns
        table[floats] = table[floats].round(2)
        with st.expander("Table view", expanded=False):
            st.dataframe(table, hide_index=True)


# --------------------------------------------------------------------------- figures


def rolling(df: pd.DataFrame, col: str, date_col: str = "local_date") -> pd.DataFrame:
    """Daily series on a full calendar index with 7- and 28-day rolling means."""
    s = df.set_index(date_col)[col].astype(float).sort_index()
    s = s[~s.index.duplicated()].asfreq("D")
    out = pd.DataFrame({"value": s})
    out["r7"] = s.rolling("7D", min_periods=3).mean()
    out["r28"] = s.rolling("28D", min_periods=10).mean()
    return out.reset_index().rename(columns={"index": date_col})


def trend(
    df: pd.DataFrame,
    col: str,
    unit: str,
    height: int = 300,
    fmt: str = ",.0f",
    show_r7: bool = True,
    clip: tuple | None = None,
    pal: Palette | None = None,
) -> go.Figure:
    """Daily values as quiet dots, 28-day mean as the emphasized line.

    Pass a frame that starts before the period and `clip=(start, end)` so the
    rolling means are already warmed up on the first visible day.
    """
    pal = pal or palette()
    r = rolling(df, col)
    if clip is not None:
        r = r[(r["local_date"] >= pd.Timestamp(clip[0])) & (r["local_date"] <= pd.Timestamp(clip[1]))]
    fig = go.Figure()
    fig.add_scatter(
        x=r["local_date"],
        y=r["value"],
        mode="markers",
        name="Daily",
        marker=dict(size=4, color=pal.muted, opacity=0.35),
        hovertemplate=f"%{{y:{fmt}}} {unit}<extra>Daily</extra>",
    )
    if show_r7:
        fig.add_scatter(
            x=r["local_date"],
            y=r["r7"],
            mode="lines",
            name="7-day mean",
            line=dict(width=1.5, color=SEQ_RAMP[3] if pal.mode == "light" else SEQ_RAMP[9]),
            hovertemplate=f"%{{y:{fmt}}} {unit}<extra>7-day</extra>",
        )
    fig.add_scatter(
        x=r["local_date"],
        y=r["r28"],
        mode="lines",
        name="28-day mean",
        line=dict(width=2.5, color=pal.series[0]),
        hovertemplate=f"%{{y:{fmt}}} {unit}<extra>28-day</extra>",
    )
    style(fig, height=height, pal=pal)
    fig.update_yaxes(title_text=unit)
    return fig


def weekday_hour_heatmap(
    df: pd.DataFrame,
    value_col: str,
    label: str,
    fmt: str = ",.0f",
    height: int = 300,
    fill: float | None = None,
    pal: Palette | None = None,
) -> go.Figure:
    """Mon..Sun x 0..23 grid from rows with Spark day_of_week (1=Sun..7=Sat)."""
    pal = pal or palette()
    order = [2, 3, 4, 5, 6, 7, 1]
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grid = df.pivot_table(index="day_of_week", columns="local_hour", values=value_col).reindex(
        index=order, columns=range(24)
    )
    if fill is not None:
        grid = grid.fillna(fill)
    fig = go.Figure(
        go.Heatmap(
            z=grid.values,
            x=[f"{h:02d}" for h in range(24)],
            y=names,
            colorscale=seq_scale(pal),
            xgap=2,
            ygap=2,
            hoverongaps=False,
            colorbar=dict(
                title=dict(text=label, side="right", font=dict(color=pal.ink2)),
                tickfont=dict(color=pal.muted),
                outlinewidth=0,
                thickness=10,
            ),
            hovertemplate=f"%{{y}} %{{x}}:00<br>{label}: %{{z:{fmt}}}<extra></extra>",
        )
    )
    style(fig, height=height, legend=False, pal=pal)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="Hour of day")
    return fig
