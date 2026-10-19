"""ECG signal processing: band-pass filter, R-peak detection, beat segmentation, HRV."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

BEAT_BEFORE_S = 0.25
BEAT_AFTER_S = 0.45
BEAT_LEN = 256


def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0, order: int = 3) -> np.ndarray:
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def detect_r_peaks(x: np.ndarray, fs: float, skip_s: float = 1.0) -> tuple[np.ndarray, int]:
    """R peaks on a band-passed single-lead ECG. Returns (sample indices, polarity +1/-1).

    Pan-Tompkins style: squared derivative, 120 ms moving-window integration, peaks at
    least 0.3 s apart (max 200 bpm) above 30% of the typical beat energy (median of the 30
    strongest candidates, robust to a few motion artifacts). The first ``skip_s`` seconds
    (filter / electrode settling transient) are ignored. Each R is then placed on the
    dominant deflection of the raw signal; the polarity is flipped when that deflection
    is negative (watch on the other wrist / reversed contact).
    """
    energy = np.convolve(np.diff(x, prepend=x[0]) ** 2, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode="same")
    energy[: int(skip_s * fs)] = 0
    cand, props = find_peaks(energy, distance=int(0.3 * fs), height=0)
    if len(cand) == 0:
        return cand, 1
    typical = np.median(np.sort(props["peak_heights"])[-30:])
    det = cand[props["peak_heights"] > 0.3 * typical]
    half = int(0.08 * fs)
    wins = [(max(0, d - half), min(len(x), d + half)) for d in det]
    up = np.median([x[a:b].max() for a, b in wins])
    down = np.median([-x[a:b].min() for a, b in wins])
    polarity = 1 if up >= down else -1
    peaks = np.array(sorted({a + int(np.argmax(polarity * x[a:b])) for a, b in wins}), dtype=int)
    if len(peaks) > 1:  # drop the weaker of two peaks closer than 0.3 s after re-alignment
        keep = [peaks[0]]
        for pk in peaks[1:]:
            if pk - keep[-1] < 0.3 * fs:
                if polarity * x[pk] > polarity * x[keep[-1]]:
                    keep[-1] = pk
            else:
                keep.append(pk)
        peaks = np.array(keep, dtype=int)
    return peaks, polarity


def segment_beats(x: np.ndarray, peaks: np.ndarray, fs: float, length: int = BEAT_LEN):
    """Fixed-length beats around each R peak, resampled to ``length`` samples (handles 512 vs 513 Hz)."""
    before, after = int(BEAT_BEFORE_S * fs), int(BEAT_AFTER_S * fs)
    keep = [p for p in peaks if p - before >= 0 and p + after < len(x)]
    grid = np.linspace(0, before + after, length)
    beats = (
        np.stack([np.interp(grid, np.arange(before + after + 1), x[p - before : p + after + 1]) for p in keep])
        if keep
        else np.zeros((0, length))
    )
    return beats, np.array(keep, dtype=int)


def hrv_from_peaks(peaks: np.ndarray, fs: float) -> dict:
    """HR, SDNN and RMSSD from RR intervals after removing implausible / ectopic-looking ones
    (outside 300-2000 ms or more than 25% away from the recording's median RR)."""
    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr):
        med = np.median(rr)
        rr = rr[np.abs(rr - med) <= 0.25 * med]
    if len(rr) < 3:
        return {"hr_bpm": np.nan, "sdnn_ms": np.nan, "rmssd_ms": np.nan, "n_rr": len(rr)}
    return {
        "hr_bpm": float(60000.0 / rr.mean()),
        "sdnn_ms": float(rr.std(ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(np.diff(rr) ** 2))),
        "n_rr": len(rr),
    }
