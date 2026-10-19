import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import data
import ui

st.title("ECG")
st.caption(
    "Single-lead (Lead I) recordings from the Apple Watch ECG app. Not a medical device readout. "
    "All recordings are listed regardless of the period filter."
)
pal = charts.palette()

ecgs = data.ecg_list()
if ecgs.empty:
    ui.empty("No ECG recordings in the export.")
    st.stop()


def r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Simple R-peak detector: baseline removal, amplitude threshold, 300 ms refractory period."""
    win = max(1, int(0.6 * fs))
    baseline = np.convolve(x, np.ones(win) / win, mode="same")
    y = x - baseline
    if -np.percentile(y, 0.5) > np.percentile(y, 99.5):  # inverted polarity
        y = -y
    thr = 0.5 * np.percentile(y, 99.5)
    cand = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > thr))[0] + 1
    refractory = int(0.3 * fs)
    peaks: list[int] = []
    for i in cand:
        if not peaks or i - peaks[-1] > refractory:
            peaks.append(int(i))
        elif y[i] > y[peaks[-1]]:
            peaks[-1] = int(i)
    return np.asarray(peaks, dtype=int)


labels = {r.ecg_id: f"{r.recorded_local:%d %b %Y, %H:%M} · {r.classification}" for r in ecgs.itertuples()}
by_label = {v: k for k, v in labels.items()}
ecg_id = by_label[st.selectbox("Recording", list(by_label), key="ecg_id")]
meta = ecgs[ecgs["ecg_id"] == ecg_id].iloc[0]
samples = data.ecg_samples(ecg_id)
fs = float(meta["sample_rate_hz"])

status_ok = str(meta["classification"]).lower() == "sinus rhythm"
st.badge(
    str(meta["classification"]),
    icon=":material/check_circle:" if status_ok else ":material/help:",
    color="green" if status_ok else "orange",
)

if samples.empty:
    ui.empty("No samples stored for this recording.")
    st.stop()

x = samples["value_uv"].to_numpy(dtype=float)
t = samples["sample_index"].to_numpy() / fs
peaks = r_peaks(x, fs)
rr = np.diff(peaks) / fs
bpm = 60 / np.median(rr) if len(rr) >= 3 else None

cols = st.columns(5)
cols[0].metric(
    "Heart rate (estimated)",
    f"{bpm:.0f} bpm" if bpm else "–",
    border=True,
    help="60 / median R-R interval from a simple peak detector",
)
cols[1].metric("Beats detected", f"{len(peaks)}", border=True)
cols[2].metric(
    "R-R variability (SDNN)",
    f"{np.std(rr) * 1000:.0f} ms" if len(rr) >= 3 else "–",
    border=True,
    help="Short 30 s window; not comparable with the Watch's daily HRV",
)
cols[3].metric("Duration", f"{len(x) / fs:.0f} s", border=True)
cols[4].metric("Sample rate", f"{fs:.0f} Hz", border=True)
if pd.notna(meta.get("symptoms")) and str(meta["symptoms"]).strip():
    st.caption(f"Symptoms noted: {meta['symptoms']}")

total = float(t[-1]) if len(t) else 0.0
window = 10.0
t0 = (
    st.slider(
        "Window start (s)",
        0.0,
        max(0.0, total - window),
        0.0,
        step=1.0,
        key="ecg_t0",
        help="Moves a 10-second window through the 30-second recording",
    )
    if total > window
    else 0.0
)
sel = (t >= t0) & (t <= t0 + window)
pk = peaks[(t[peaks] >= t0) & (t[peaks] <= t0 + window)] if len(peaks) else peaks

fig = go.Figure()
fig.add_trace(
    go.Scattergl(
        x=t[sel],
        y=x[sel] / 1000,
        mode="lines",
        name="Lead I",
        line=dict(color=pal.series[0], width=1.2),
        hovertemplate="%{x:.2f} s: %{y:.2f} mV<extra></extra>",
    )
)
if len(pk):
    fig.add_trace(
        go.Scattergl(
            x=t[pk],
            y=x[pk] / 1000,
            mode="markers",
            name="R peak",
            marker=dict(size=8, color=pal.series[1], line=dict(color=pal.surface, width=2)),
            hovertemplate="R peak %{x:.2f} s<extra></extra>",
        )
    )
charts.style(fig, height=340, pal=pal)
fig.update_layout(hovermode="closest")
lo_mv, hi_mv = x.min() / 1000, x.max() / 1000
fig.update_xaxes(title_text="seconds", range=[t0, t0 + window], dtick=1, showgrid=True)
fig.update_yaxes(title_text="mV", dtick=0.5, range=[lo_mv - 0.2, hi_mv + 0.2])
charts.show(fig, key="ecg_wave")

if len(rr) >= 3:
    left, right = st.columns(2)
    with left:
        st.markdown("**Beat-to-beat interval**")
        rf = go.Figure(
            go.Scatter(
                x=t[peaks[1:]],
                y=rr * 1000,
                mode="lines+markers",
                name="R-R",
                line=dict(color=pal.series[0], width=2),
                marker=dict(size=6),
                hovertemplate="%{x:.1f} s: %{y:.0f} ms<extra></extra>",
            )
        )
        charts.style(rf, height=240, legend=False, pal=pal)
        rf.update_xaxes(title_text="seconds")
        rf.update_yaxes(title_text="R-R (ms)")
        charts.show(rf, key="ecg_rr")
    with right:
        st.markdown("**Average beat**")
        half = int(0.35 * fs)
        beats = np.array([x[i - half : i + half] for i in peaks if i - half >= 0 and i + half <= len(x)])
        if len(beats):
            bt = (np.arange(-half, half) / fs) * 1000
            bf = go.Figure()
            for b in beats[:40]:
                bf.add_scatter(
                    x=bt,
                    y=b / 1000,
                    mode="lines",
                    line=dict(color=pal.muted, width=1),
                    opacity=0.25,
                    hoverinfo="skip",
                    showlegend=False,
                )
            bf.add_scatter(
                x=bt,
                y=beats.mean(axis=0) / 1000,
                mode="lines",
                name="Mean beat",
                line=dict(color=pal.series[0], width=2.5),
                hovertemplate="%{x:.0f} ms: %{y:.2f} mV<extra></extra>",
            )
            charts.style(bf, height=240, legend=False, pal=pal)
            bf.update_xaxes(title_text="ms from R peak")
            bf.update_yaxes(title_text="mV")
            charts.show(bf, key="ecg_beat")

st.subheader("All recordings")
st.dataframe(
    ecgs[["recorded_local", "classification", "symptoms", "device", "duration_s", "std_uv", "min_uv", "max_uv"]],
    hide_index=True,
    column_config={
        "recorded_local": st.column_config.DatetimeColumn("Recorded", format="DD MMM YYYY, HH:mm"),
        "classification": "Classification",
        "symptoms": "Symptoms",
        "device": "Device",
        "duration_s": st.column_config.NumberColumn("Seconds", format="%.0f"),
        "std_uv": st.column_config.NumberColumn("Std (µV)", format="%.0f"),
        "min_uv": st.column_config.NumberColumn("Min (µV)", format="%.0f"),
        "max_uv": st.column_config.NumberColumn("Max (µV)", format="%.0f"),
    },
)
