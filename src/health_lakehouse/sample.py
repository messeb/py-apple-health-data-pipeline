"""Synthetic Apple Health export for demos, tests and CI.

`generate()` writes a folder that is laid out like the one the iPhone Health app
exports (`export.xml`, `export_cda.xml`, `workout-routes/*.gpx`,
`electrocardiograms/*.csv`, `clinical-records/*.json`), so the real pipeline
runs on it unchanged:

    health-lakehouse sample --out sample_data/apple_health_export --days 730 --seed 42

The person is fictional ("Jane Appleseed") and every value is drawn from a
seeded random generator; nothing is derived from a real export. The signals
are built to give the app and the ML experiments something to find:

* weekly rhythm (more steps on weekdays, longer workouts at weekends, resting
  HR a little higher on Monday/Tuesday), seasonality (more steps and cycling
  in summer) and a slow fitness trend (resting HR falls, HRV and VO2max rise
  while the training load grows),
* a few injected illness episodes (resting HR up, HRV down, steps down, no
  workouts for 3-5 days) for the anomaly detector,
* days without the watch (phone-only data) and one longer watch gap,
* Apple Watch and iPhone both counting steps/distance (exercises the dedup),
* a newer sensor (sleeping wrist temperature) that only appears late.

Like Apple, every timestamp in `export.xml` carries the UTC offset that was
active at *export* time, even though the local times follow DST.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

__all__ = ["DEFAULT_END", "PERSON", "generate"]

DEFAULT_END = date(2026, 9, 30)  # fixed, so a seed always produces the same export
PERSON = "Jane Appleseed"
HOME = (52.4106, 12.0153)  # arbitrary rural spot; routes loop around it
HEIGHT_M = 1.68

WATCH = "Jane's Apple Watch"
PHONE = "Jane's iPhone"
SCALE = "SmartScale"
CUFF = "BP Monitor"

HK_Q = "HKQuantityTypeIdentifier"
HK_C = "HKCategoryTypeIdentifier"
SLEEP = {
    "in_bed": "HKCategoryValueSleepAnalysisInBed",
    "core": "HKCategoryValueSleepAnalysisAsleepCore",
    "deep": "HKCategoryValueSleepAnalysisAsleepDeep",
    "rem": "HKCategoryValueSleepAnalysisAsleepREM",
    "awake": "HKCategoryValueSleepAnalysisAwake",
}

# Diurnal step profiles (share of the day's steps per local hour).
_WEEKDAY_STEPS = np.array([0, 0, 0, 0, 0, 0, 1, 6, 10, 4, 3, 4, 8, 4, 3, 4, 5, 9, 8, 5, 4, 3, 1, 0], float)
_WEEKEND_STEPS = np.array([0, 0, 0, 0, 0, 0, 0, 1, 3, 6, 8, 9, 8, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0], float)


@dataclass(frozen=True)
class WorkoutKind:
    hk: str
    minutes: float
    hr: float
    mets: float
    speed_kmh: float = 0.0  # 0 = no distance
    distance_type: str | None = None
    outdoor: bool = False
    stride_m: float = 0.0  # > 0: the workout adds steps


KINDS = {
    "Cycling": WorkoutKind("Cycling", 60, 132, 7.5, 22, "DistanceCycling", True),
    "Walking": WorkoutKind("Walking", 45, 104, 3.8, 5.3, "DistanceWalkingRunning", True, 0.74),
    "Running": WorkoutKind("Running", 35, 152, 9.8, 10, "DistanceWalkingRunning", True, 1.05),
    "Hiking": WorkoutKind("Hiking", 150, 114, 6.0, 4.2, "DistanceWalkingRunning", True, 0.70),
    "Swimming": WorkoutKind("Swimming", 40, 126, 7.0, 2.2, "DistanceSwimming"),
    "TraditionalStrengthTraining": WorkoutKind("TraditionalStrengthTraining", 45, 112, 5.0),
    "Yoga": WorkoutKind("Yoga", 30, 92, 3.0),
}


@dataclass(frozen=True)
class Source:
    name: str
    version: str
    device: str | None = None

    def attrs(self) -> str:
        dev = f' device="{self.device}"' if self.device else ""
        return f'sourceName="{self.name}" sourceVersion="{self.version}"{dev}'


def _device(name: str, model: str, hardware: str, software: str, created: str, ptr: int) -> str:
    return (
        f"&lt;&lt;HKDevice: 0x{ptr:010x}&gt;, name:{name}, manufacturer:Apple Inc., model:{model}, "
        f"hardware:{hardware}, software:{software}, creation date:{created} +0000&gt;"
    )


class _Clock:
    """Converts true local wall times to Apple's export-offset timestamp strings."""

    def __init__(self, tz: str, export_local: datetime):
        self.tz = ZoneInfo(tz)
        self.export_offset = export_local.replace(tzinfo=self.tz).utcoffset()
        minutes = int(self.export_offset.total_seconds() // 60)
        sign = "+" if minutes >= 0 else "-"
        self.suffix = f" {sign}{abs(minutes) // 60:02d}{abs(minutes) % 60:02d}"
        self._offsets: dict[tuple[date, int], timedelta] = {}

    def offset(self, local: datetime) -> timedelta:
        key = (local.date(), local.hour)
        off = self._offsets.get(key)
        if off is None:
            off = self._offsets[key] = local.replace(tzinfo=self.tz).utcoffset()
        return off

    def utc(self, local: datetime) -> datetime:
        return local - self.offset(local)

    def stamp(self, local: datetime) -> str:
        """Apple format `YYYY-MM-DD HH:MM:SS +ZZZZ`, always with the export-time offset."""
        return (self.utc(local) + self.export_offset).strftime("%Y-%m-%d %H:%M:%S") + self.suffix

    def true_stamp(self, local: datetime) -> str:
        """Same format with the offset active at that moment (used in ECG headers)."""
        minutes = int(self.offset(local).total_seconds() // 60)
        sign = "+" if minutes >= 0 else "-"
        return local.strftime("%Y-%m-%d %H:%M:%S") + f" {sign}{abs(minutes) // 60:02d}{abs(minutes) % 60:02d}"


def _num(v: float, digits: int = 4) -> str:
    s = f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


@dataclass
class _Export:
    clock: _Clock
    lines: dict[str, list[str]] = field(default_factory=dict)
    workouts: list[str] = field(default_factory=list)
    correlations: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        hk_type: str,
        src: Source,
        start: datetime,
        end: datetime,
        value,
        unit: str | None = None,
        meta: dict | None = None,
        children: list[str] | None = None,
        created: datetime | None = None,
        top_level: bool = True,
    ) -> str:
        c = self.clock
        unit_attr = f' unit="{unit}"' if unit else ""
        value = value if isinstance(value, str) else _num(value)
        head = (
            f'<Record type="{hk_type}" {src.attrs()}{unit_attr} creationDate="{c.stamp(created or end)}" '
            f'startDate="{c.stamp(start)}" endDate="{c.stamp(end)}" value="{value}"'
        )
        body = [f'  <MetadataEntry key="{k}" value="{v}"/>' for k, v in (meta or {}).items()] + (children or [])
        xml = f" {head}/>" if not body else "\n".join([f" {head}>", *body, " </Record>"])
        if top_level:
            self.lines.setdefault(hk_type, []).append(xml)
            self.counts[hk_type] = self.counts.get(hk_type, 0) + 1
        return xml

    def xml_records(self) -> list[str]:
        return [line for t in self.lines for line in self.lines[t]]


@dataclass
class _Day:
    """The latent daily state the observations are generated from."""

    day: date
    i: int
    trend: float  # 0 -> 1 over the period (fitness)
    season: float  # +1 mid-summer, -1 mid-winter
    weekday: int  # 0 = Monday
    illness: float  # 0 healthy .. 1 worst day of an episode
    watch: bool  # watch worn
    rhr: float
    hrv: float
    steps: float
    weight: float
    resp: float = 14.5  # nightly respiratory rate level (breaths/min)
    spo2: float = 0.972  # nightly SpO2 level (fraction)
    fever: float = 0.0  # illness: wrist temperature / HR elevation


def _daily_state(rng: np.random.Generator, start: date, n: int) -> list[_Day]:
    days = [start + timedelta(days=i) for i in range(n)]
    trend = np.linspace(0, 1, n)
    doy = np.array([d.timetuple().tm_yday for d in days])
    season = np.cos(2 * np.pi * (doy - 196) / 365.25)
    weekday = np.array([d.weekday() for d in days])

    # Watch not worn: ~2% random days plus one 6-day gap (charger forgotten on holiday).
    watch = rng.random(n) > 0.02
    if n > 120:
        g = int(rng.integers(int(n * 0.35), int(n * 0.45)))
        watch[g : g + 6] = False
    watch[:3] = True

    # Illness episodes: one per ~150 days, 3-5 days, never inside a watch gap. Each episode
    # has its own mix of symptoms, so they are not one repeated pattern a model could learn.
    illness, rhr_add, hrv_keep, steps_keep = np.zeros(n), np.zeros(n), np.ones(n), np.ones(n)
    resp_add, spo2_drop, fever = np.zeros(n), np.zeros(n), np.zeros(n)
    n_ep = max(1, n // 150)
    for k in range(n_ep):
        lo, hi = int(n * k / n_ep) + 20, int(n * (k + 1) / n_ep) - 20
        if hi - lo < 6:
            continue
        s = int(rng.integers(lo, hi - 5))
        length = int(rng.integers(3, 6))
        shape = np.sin(np.linspace(0.6, np.pi - 0.4, length))
        sl = slice(s, s + length)
        illness[sl] = np.maximum(illness[sl], shape)
        rhr_add[sl] += rng.uniform(7, 14) * shape
        hrv_keep[sl] *= 1 - rng.uniform(0.3, 0.55) * shape
        steps_keep[sl] *= 1 - rng.uniform(0.5, 0.85) * shape
        resp_add[sl] += rng.uniform(1.0, 3.5) * shape
        spo2_drop[sl] += rng.uniform(0.005, 0.03) * shape
        fever[sl] += rng.uniform(0.4, 1.5) * shape
        watch[sl] = True

    # Slow AR(1) wanders so consecutive days are correlated (forecastable); RHR and HRV
    # wander mostly independently, only illness moves them together.
    ar, ar_hrv, ar_resp = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 0.6)
        ar_hrv[i] = 0.75 * ar_hrv[i - 1] + rng.normal(0, 1.6)
        ar_resp[i] = 0.7 * ar_resp[i - 1] + rng.normal(0, 0.35)
    resp = 14.5 + 0.3 * season + ar_resp + rng.normal(0, 0.3, n) + resp_add
    spo2 = np.clip(0.972 + rng.normal(0, 0.006, n) - spo2_drop, 0.9, 0.995)

    dow_rhr = np.array([1.4, 1.0, 0.2, 0.0, -0.3, -0.9, -0.6])
    rhr = 63 - 6 * trend - 1.2 * season + dow_rhr[weekday] + ar + rhr_add + rng.normal(0, 0.9, n)
    hrv = (36 + 11 * trend + 2.5 * season - 0.8 * dow_rhr[weekday] - 0.3 * ar + ar_hrv) * hrv_keep
    hrv = np.clip(hrv * rng.lognormal(0, 0.08, n), 12, None)
    dow_steps = np.array([1.10, 1.07, 1.08, 1.06, 1.04, 0.76, 0.68])
    steps = 8200 * dow_steps[weekday] * (1 + 0.17 * season + 0.08 * trend) * steps_keep
    steps = steps * rng.lognormal(0, 0.18, n)
    weight = 70.5 - 3.0 * trend + 0.6 * np.cos(2 * np.pi * (doy - 15) / 365.25) + np.cumsum(rng.normal(0, 0.03, n))
    return [
        _Day(
            days[i],
            i,
            float(trend[i]),
            float(season[i]),
            int(weekday[i]),
            float(illness[i]),
            bool(watch[i]),
            float(rhr[i]),
            float(hrv[i]),
            float(steps[i]),
            float(weight[i]),
            float(resp[i]),
            float(spo2[i]),
            float(fever[i]),
        )
        for i in range(n)
    ]


def _at(d: date, minutes: float) -> datetime:
    """Local wall time `minutes` after midnight of `d` (may run into the next day)."""
    return datetime(d.year, d.month, d.day) + timedelta(minutes=float(minutes))


@dataclass
class _Night:
    bed: float  # minutes after midnight of the evening's date
    wake: float
    stages: list[tuple[str, float, float]]  # (stage, start_min, end_min)


def _sleep(rng: np.random.Generator, d: _Day) -> _Night:
    late = d.weekday in (4, 5)  # Friday / Saturday night
    bed = (24 * 60 + 15 if late else 23 * 60 + 10) + rng.normal(0, 35 if late else 22) - 50 * d.illness
    asleep = 7 * 60 + rng.normal(0, 35) + (40 if late else 0) + 70 * d.illness
    t = bed + rng.uniform(5, 20)  # falling asleep
    stages: list[tuple[str, float, float]] = []
    cycle = 0
    while t < bed + asleep:
        deep = max(3.0, rng.normal(24 - 6 * cycle, 5) * (1 + 0.15 * d.trend))
        rem = max(4.0, rng.normal(8 + 5 * cycle, 4))
        for stage, length in (("core", rng.normal(38, 8)), ("deep", deep), ("core", rng.normal(14, 4)), ("rem", rem)):
            length = float(max(3.0, length))
            stages.append((stage, t, t + length))
            t += length
        if rng.random() < 0.55:
            gap = float(rng.uniform(1, 6))
            stages.append(("awake", t, t + gap))
            t += gap
        cycle += 1
    return _Night(bed=float(bed), wake=float(t + rng.uniform(2, 10)), stages=stages)


@dataclass
class _Workout:
    kind: WorkoutKind
    start: float  # minutes after midnight
    minutes: float
    hr_avg: float
    hr_min: float
    hr_max: float
    distance_km: float
    active_kcal: float
    basal_kcal: float
    indoor: bool
    temp_c: float
    humidity: float
    elevation_m: float
    route: str | None = None


def _plan_workouts(rng: np.random.Generator, d: _Day) -> list[_Workout]:
    if not d.watch or d.illness > 0:
        return []
    weekend = d.weekday >= 5
    p = [0.45, 0.55, 0.45, 0.55, 0.3, 0.7, 0.65][d.weekday] * (0.75 + 0.45 * d.trend)
    if rng.random() > min(p, 0.95):
        return []
    s = d.season
    names = ["Cycling", "Walking", "Running", "Swimming", "TraditionalStrengthTraining", "Yoga", "Hiking"]
    weights = np.array(
        [
            0.30 + 0.2 * s,
            0.22,
            0.14 + 0.05 * d.trend,
            0.08 - 0.04 * s,
            0.13 - 0.03 * s,
            0.08,
            0.06 if weekend and s > -0.3 else 0.0,
        ]
    )
    weights = np.clip(weights, 0.01, None)
    plan = [names[int(rng.choice(len(names), p=weights / weights.sum()))]]
    if weekend and rng.random() < 0.15 and plan[0] != "Walking":
        plan.append("Walking")
    out = []
    t = (
        rng.normal(10 * 60, 90)
        if weekend
        else (rng.normal(7 * 60, 20) if rng.random() < 0.25 else rng.normal(18 * 60, 50))
    )
    for name in plan:
        k = KINDS[name]
        minutes = k.minutes * (1.5 if weekend and name != "Hiking" else 1.0) * (1 + 0.3 * d.trend)
        minutes = float(np.clip(minutes * rng.lognormal(0, 0.22), 8, 330))
        indoor = not k.outdoor or (name == "Cycling" and s < -0.3 and rng.random() < 0.5)
        speed = (
            k.speed_kmh * (1 + 0.08 * d.trend) * rng.lognormal(0, 0.07) * (0.8 if indoor and name == "Cycling" else 1)
        )
        hr_avg = k.hr - 5 * d.trend + (d.rhr - 60) * 0.5 + rng.normal(0, 5)
        mets = k.mets * rng.lognormal(0, 0.08)
        out.append(
            _Workout(
                kind=k,
                start=float(np.clip(t, 5 * 60, 21 * 60)),
                minutes=minutes,
                hr_avg=hr_avg,
                hr_min=max(d.rhr + 15, hr_avg - rng.uniform(25, 45)),
                hr_max=hr_avg + rng.uniform(15, 32),
                distance_km=speed * minutes / 60 if k.distance_type else 0.0,
                active_kcal=(mets - 1) * d.weight * minutes / 60,
                basal_kcal=1.0 * d.weight * minutes / 60,
                indoor=indoor,
                temp_c=float(11 + 9 * s + rng.normal(0, 4)),
                humidity=float(np.clip(rng.normal(70 - 8 * s, 12), 25, 99)),
                elevation_m=float(0 if indoor or not k.distance_type else speed * minutes / 60 * rng.uniform(4, 14)),
            )
        )
        t += minutes + rng.uniform(60, 180)
    return out


def _spread(arr: np.ndarray, start_min: float, minutes: float, total: float) -> None:
    """Add `total` to the hour buckets an interval overlaps, proportional to the overlap."""
    end = min(start_min + minutes, 24 * 60 - 1)
    t = start_min
    while t < end:
        h = int(t // 60)
        nxt = min((h + 1) * 60, end)
        arr[h] += total * (nxt - t) / max(minutes, 1e-9)
        t = nxt


def _route_name(local: datetime) -> str:
    h12 = local.hour % 12 or 12
    return f"route_{local:%Y-%m-%d}_{h12}.{local.minute:02d}{'am' if local.hour < 12 else 'pm'}.gpx"


def _gpx(rng: np.random.Generator, w: _Workout, start_utc: datetime, step_s: int = 15) -> str:
    """A closed loop around HOME whose length matches the workout distance."""
    n = max(4, int(w.minutes * 60 / step_s))
    radius_km = w.distance_km / (2 * math.pi * 1.12)
    rot, phase = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
    centre = (HOME[0] + rng.normal(0, 0.004), HOME[1] + rng.normal(0, 0.006))
    km_lat, km_lon = 1 / 110.574, 1 / (111.32 * math.cos(math.radians(HOME[0])))
    theta = np.linspace(0, 2 * math.pi, n) + rot
    r = radius_km * (1 + 0.22 * np.sin(3 * theta + phase) + 0.08 * np.sin(7 * theta))
    lat = centre[0] + r * np.sin(theta) * km_lat + rng.normal(0, 2e-6, n)
    lon = centre[1] + r * np.cos(theta) * km_lon + rng.normal(0, 2e-6, n)
    ele = 48 + 9 * np.sin(2 * theta + phase) + 3 * np.sin(5 * theta) + rng.normal(0, 0.3, n)
    speed = np.clip(w.distance_km / (w.minutes / 60) / 3.6 * rng.lognormal(0, 0.12, n), 0, None)
    course = (np.degrees(np.arctan2(np.gradient(lon) / km_lon, np.gradient(lat) / km_lat)) + 360) % 360
    pts = []
    for i in range(n):
        ts = (start_utc + timedelta(seconds=i * step_s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append(
            f'      <trkpt lon="{lon[i]:.6f}" lat="{lat[i]:.6f}"><ele>{ele[i]:.6f}</ele><time>{ts}</time>'
            f"<extensions><speed>{speed[i]:.6f}</speed><course>{course[i]:.6f}</course>"
            f"<hAcc>{rng.uniform(1.5, 8):.6f}</hAcc><vAcc>{rng.uniform(1, 5):.6f}</vAcc></extensions></trkpt>"
        )
    name = w.route[len("route_") : -len(".gpx")] if w.route else ""
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="Apple Health Export" xmlns="http://www.topografix.com/GPX/1/1" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">',
            "  <metadata>",
            f"    <time>{start_utc:%Y-%m-%dT%H:%M:%SZ}</time>",
            "  </metadata>",
            "  <trk>",
            f"    <name>Route {name.replace('_', ' ')}</name>",
            "    <trkseg>",
            *pts,
            "    </trkseg>",
            "  </trk>",
            "</gpx>",
            "",
        ]
    )


# --------------------------------------------------------------------------- ECG


def ecg_signal(
    rng: np.random.Generator,
    fs: float = 512.0,
    seconds: float = 30.0,
    hr: float = 68.0,
    rhythm: str = "sinus",
    noise_uv: float = 12.0,
) -> np.ndarray:
    """Lead-I-like trace in µV: Gaussian P-QRS-T waves, baseline wander, mains hum and noise.

    `rhythm="afib"` drops the P wave, adds fibrillatory waves and draws irregular RR intervals.
    """
    t = np.arange(int(fs * seconds)) / fs
    x = 45 * np.sin(2 * np.pi * 0.28 * t + rng.uniform(0, 6)) + 4 * np.sin(2 * np.pi * 50 * t)
    x += rng.normal(0, noise_uv, len(t))
    rr_mean = 60 / hr
    beats, b = [], rng.uniform(0.2, 0.6)
    while b < seconds - 0.3:
        beats.append(b)
        if rhythm == "afib":
            b += float(np.clip(rng.normal(rr_mean * 0.85, rr_mean * 0.25), 0.33, 1.4))
        else:
            b += rr_mean * (1 + 0.04 * math.sin(2 * math.pi * b / 4.5)) + rng.normal(0, 0.012)
    waves = [(-0.03, 0.010, -110), (0.0, 0.011, 1050), (0.032, 0.012, -260), (0.26, 0.045, 300)]
    if rhythm != "afib":
        waves.append((-0.17, 0.022, 130))
    else:
        x += 28 * np.sin(2 * np.pi * rng.uniform(5, 7) * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.7 * t))
    for b in beats:
        amp = rng.normal(1, 0.03)
        lo, hi = max(0, int((b - 0.4) * fs)), min(len(t), int((b + 0.6) * fs))
        seg = t[lo:hi] - b
        for off, width, a in waves:
            x[lo:hi] += amp * a * np.exp(-((seg - off) ** 2) / (2 * width**2))
    return x


def _ecg_csv(signal: np.ndarray, recorded: str, classification: str, symptoms: str, german: bool) -> str:
    if german:
        dob, rate = '"1. Apr. 1990"', "512,000 hertz"
        values = [f"{v:.3f}".replace(".", ",") for v in signal]
    else:
        dob, rate = '"Apr 1, 1990"', "512.000 hertz"
        values = [f"{v:.3f}" for v in signal]
    header = [
        f"Name,{PERSON}",
        f"Date of Birth,{dob}",
        f"Recorded Date,{recorded}",
        f"Classification,{classification}",
        f"Symptoms,{symptoms}",
        "Software Version,2.0",
        'Device,"Watch7,3"',
        f"Sample Rate,{rate}",
        "",
        "",
        "Lead,Lead I",
        "Unit,µV",
        "",
    ]
    return "\n".join(header + values) + "\n"


# --------------------------------------------------------------------------- export.xml skeleton

_DTD = """<!DOCTYPE HealthData [
<!-- HealthKit Export Version: 17 -->
<!ELEMENT HealthData (ExportDate,Me,(Record|Correlation|Workout|ActivitySummary|ClinicalRecord|Audiogram|VisionPrescription)*)>
<!ATTLIST HealthData
  locale CDATA #REQUIRED
>
<!ELEMENT ExportDate EMPTY>
<!ATTLIST ExportDate
  value CDATA #REQUIRED
>
<!ELEMENT Me EMPTY>
<!ATTLIST Me
  HKCharacteristicTypeIdentifierDateOfBirth                   CDATA #REQUIRED
  HKCharacteristicTypeIdentifierBiologicalSex                 CDATA #REQUIRED
  HKCharacteristicTypeIdentifierBloodType                     CDATA #REQUIRED
  HKCharacteristicTypeIdentifierFitzpatrickSkinType           CDATA #REQUIRED
  HKCharacteristicTypeIdentifierCardioFitnessMedicationsUse   CDATA #REQUIRED
>
<!ELEMENT Record ((MetadataEntry|HeartRateVariabilityMetadataList)*)>
<!ATTLIST Record
  type          CDATA #REQUIRED
  unit          CDATA #IMPLIED
  value         CDATA #IMPLIED
  sourceName    CDATA #REQUIRED
  sourceVersion CDATA #IMPLIED
  device        CDATA #IMPLIED
  creationDate  CDATA #IMPLIED
  startDate     CDATA #REQUIRED
  endDate       CDATA #REQUIRED
>
<!-- Note: Any Records that appear as children of a correlation also appear as top-level records in this document. -->
<!ELEMENT Correlation ((MetadataEntry|Record)*)>
<!ATTLIST Correlation
  type          CDATA #REQUIRED
  sourceName    CDATA #REQUIRED
  sourceVersion CDATA #IMPLIED
  device        CDATA #IMPLIED
  creationDate  CDATA #IMPLIED
  startDate     CDATA #REQUIRED
  endDate       CDATA #REQUIRED
>
<!-- Note: Any Records that appear as children of a Workout also appear as top-level records in this document. -->
<!ELEMENT Workout ((MetadataEntry|WorkoutEvent|WorkoutRoute|WorkoutStatistics|WorkoutActivity|WorkoutZoneGroup|Record)*)>
<!ATTLIST Workout
  workoutActivityType   CDATA #REQUIRED
  duration              CDATA #IMPLIED
  durationUnit          CDATA #IMPLIED
  totalDistance         CDATA #IMPLIED
  totalDistanceUnit     CDATA #IMPLIED
  totalEnergyBurned     CDATA #IMPLIED
  totalEnergyBurnedUnit CDATA #IMPLIED
  sourceName            CDATA #REQUIRED
  sourceVersion         CDATA #IMPLIED
  device                CDATA #IMPLIED
  creationDate          CDATA #IMPLIED
  startDate             CDATA #REQUIRED
  endDate               CDATA #REQUIRED
>
<!ELEMENT WorkoutEvent ((MetadataEntry)*)>
<!ATTLIST WorkoutEvent
  type                 CDATA #REQUIRED
  date                 CDATA #REQUIRED
  duration             CDATA #IMPLIED
  durationUnit         CDATA #IMPLIED
>
<!ELEMENT WorkoutStatistics EMPTY>
<!ATTLIST WorkoutStatistics
  type                 CDATA #REQUIRED
  startDate            CDATA #REQUIRED
  endDate              CDATA #REQUIRED
  average              CDATA #IMPLIED
  minimum              CDATA #IMPLIED
  maximum              CDATA #IMPLIED
  sum                  CDATA #IMPLIED
  unit                 CDATA #IMPLIED
>
<!ELEMENT WorkoutRoute ((MetadataEntry|FileReference)*)>
<!ATTLIST WorkoutRoute
  sourceName    CDATA #REQUIRED
  sourceVersion CDATA #IMPLIED
  device        CDATA #IMPLIED
  creationDate  CDATA #IMPLIED
  startDate     CDATA #REQUIRED
  endDate       CDATA #REQUIRED
>
<!ELEMENT FileReference EMPTY>
<!ATTLIST FileReference
  path CDATA #REQUIRED
>
<!ELEMENT ActivitySummary EMPTY>
<!ATTLIST ActivitySummary
  dateComponents           CDATA #IMPLIED
  activeEnergyBurned       CDATA #IMPLIED
  activeEnergyBurnedGoal   CDATA #IMPLIED
  activeEnergyBurnedUnit   CDATA #IMPLIED
  appleMoveTime            CDATA #IMPLIED
  appleMoveTimeGoal        CDATA #IMPLIED
  appleExerciseTime        CDATA #IMPLIED
  appleExerciseTimeGoal    CDATA #IMPLIED
  appleStandHours          CDATA #IMPLIED
  appleStandHoursGoal      CDATA #IMPLIED
>
<!ELEMENT MetadataEntry EMPTY>
<!ATTLIST MetadataEntry
  key   CDATA #REQUIRED
  value CDATA #REQUIRED
>
<!-- Note: Heart Rate Variability records captured by Apple Watch may include an associated list of instantaneous beats-per-minute readings. -->
<!ELEMENT HeartRateVariabilityMetadataList (InstantaneousBeatsPerMinute*)>
<!ELEMENT InstantaneousBeatsPerMinute EMPTY>
<!ATTLIST InstantaneousBeatsPerMinute
  bpm  CDATA #REQUIRED
  time CDATA #REQUIRED
>
<!ELEMENT ClinicalRecord EMPTY>
<!ATTLIST ClinicalRecord
  type              CDATA #REQUIRED
  identifier        CDATA #REQUIRED
  sourceName        CDATA #REQUIRED
  sourceURL         CDATA #REQUIRED
  fhirVersion       CDATA #REQUIRED
  receivedDate      CDATA #REQUIRED
  resourceFilePath  CDATA #REQUIRED
>
]>"""


def _uuid(rng: np.random.Generator) -> str:
    return str(uuid.UUID(bytes=rng.bytes(16), version=4)).upper()


# --------------------------------------------------------------------------- generator


class _Generator:
    def __init__(self, out: Path, days: int, seed: int, tz: str, end: date):
        self.out = out
        self.rng = np.random.default_rng(seed)
        self.tz = tz
        self.start = end - timedelta(days=days - 1)
        self.end = end
        self.n = days
        self.export_local = _at(end + timedelta(days=1), 9 * 60 + 12)
        self.clock = _Clock(tz, self.export_local)
        self.x = _Export(self.clock)
        self.german = tz.startswith("Europe/")
        self.locale = "en_DE" if self.german else "en_US"
        # Watch upgrade in the middle of the period; software versions tick up yearly.
        mid = self.start + timedelta(days=days // 2)
        self.watch_old = Source(
            WATCH,
            "10.4",
            _device("Apple Watch", "Watch", "Watch6,1", "10.4", f"{self.start:%Y-%m-%d} 07:00:00", 0x6000_1A2B_00),
        )
        self.watch_new = Source(
            WATCH,
            "11.2",
            _device("Apple Watch", "Watch", "Watch7,3", "11.2", f"{mid:%Y-%m-%d} 18:00:00", 0x6000_3C4D_00),
        )
        self.watch_switch = mid
        self.phone = Source(
            PHONE,
            "18.3",
            _device("iPhone", "iPhone", "iPhone14,7", "18.3", f"{self.start:%Y-%m-%d} 06:00:00", 0x6000_5E6F_00),
        )
        self.phone_plain = Source(PHONE, "18.3")
        self.scale = Source(SCALE, "3.4.1")
        self.cuff = Source(CUFF, "2.0.7")
        self.activity_summaries: list[str] = []
        self.routes: dict[str, str] = {}
        self.ecg_files: dict[str, str] = {}
        self.clinical: dict[str, str] = {}
        self.clinical_xml: list[str] = []

    def watch(self, d: date) -> Source:
        return self.watch_new if d >= self.watch_switch else self.watch_old

    # ----------------------------------------------------------------- per day
    def run(self) -> None:
        states = _daily_state(self.rng, self.start, self.n)
        nights = [_sleep(self.rng, s) for s in states]
        self.x.record(
            "HKDataTypeSleepDurationGoal", self.phone_plain, _at(self.start, 8 * 60), _at(self.start, 8 * 60), 8, "hr"
        )
        self.x.record(
            f"{HK_Q}Height",
            self.phone_plain,
            _at(self.start, 9 * 60),
            _at(self.start, 9 * 60),
            HEIGHT_M * 100,
            "cm",
            meta={"HKWasUserEntered": "1"},
        )
        # One week away: workouts there carry another HKTimeZone (the home zone still wins).
        vacation = set(range(int(self.n * 0.62), int(self.n * 0.62) + 7)) if self.n >= 90 else set()
        for s in states:
            prev = nights[s.i - 1] if s.i > 0 else None
            wake = (prev.wake - 24 * 60) if prev else 7 * 60
            bed = nights[s.i].bed
            self._day(s, wake, bed, "Europe/Lisbon" if s.i in vacation and self.tz != "Europe/Lisbon" else self.tz)
            self._night(s, nights[s.i], states[s.i + 1] if s.i + 1 < self.n else s)
        self._ecgs(states)
        self._clinical()

    def _day(self, s: _Day, wake: float, bed: float, tz_name: str) -> None:
        rng, x, d = self.rng, self.x, s.day
        watch = self.watch(d)
        workouts = _plan_workouts(rng, s)
        weekend = s.weekday >= 5
        prof = (_WEEKEND_STEPS if weekend else _WEEKDAY_STEPS).copy()
        hours = np.arange(24)
        awake = (hours * 60 + 59 >= wake) & (hours * 60 < min(bed, 24 * 60))
        prof = prof * awake * rng.lognormal(0, 0.35, 24)
        steps_h = s.steps * prof / max(prof.sum(), 1e-9)
        work_min_h = np.zeros(24)
        work_kcal_h = np.zeros(24)
        for w in workouts:
            _spread(work_min_h, w.start, w.minutes, w.minutes)
            _spread(work_kcal_h, w.start, w.minutes, w.active_kcal)
            if w.kind.stride_m:
                _spread(steps_h, w.start, w.minutes, w.distance_km * 1000 / w.kind.stride_m)

        # Steps + walking/running distance: watch and phone record in parallel.
        carry = rng.uniform(0.45, 0.95, 24)
        for h in range(24):
            if steps_h[h] < 15:
                continue
            m0 = rng.uniform(0, 25)
            dur = rng.uniform(8, 59 - m0)
            stride_km = rng.normal(0.00072, 0.00003)
            if s.watch:
                st, en = _at(d, h * 60 + m0), _at(d, h * 60 + m0 + dur)
                x.record(f"{HK_Q}StepCount", watch, st, en, round(steps_h[h]), "count")
                x.record(f"{HK_Q}DistanceWalkingRunning", watch, st, en, steps_h[h] * stride_km, "km")
            phone_steps = steps_h[h] * (0.85 if not s.watch else carry[h]) * (0.2 if work_min_h[h] > 20 else 1)
            if phone_steps >= 10:
                pm0 = rng.uniform(0, 20)
                st, en = _at(d, h * 60 + pm0), _at(d, h * 60 + pm0 + rng.uniform(5, 59 - pm0))
                x.record(f"{HK_Q}StepCount", self.phone, st, en, round(phone_steps), "count")
                x.record(f"{HK_Q}DistanceWalkingRunning", self.phone, st, en, phone_steps * stride_km, "km")
            if rng.random() < 0.12 * (1 + steps_h[h] / 800):
                fl = int(rng.integers(1, 4))
                st = _at(d, h * 60 + rng.uniform(0, 50))
                x.record(
                    f"{HK_Q}FlightsClimbed", self.phone, st, st + timedelta(minutes=rng.uniform(1, 5)), fl, "count"
                )

        if s.watch:
            self._watch_day(s, watch, workouts, steps_h, work_min_h, work_kcal_h, awake, wake, bed, tz_name)
        else:  # phone-only day: no rings earned
            self._summary(d, 0.0, 0, 0)
        # Gait metrics come from the iPhone.
        for _ in range(int(rng.integers(2, 6))):
            st = _at(d, rng.uniform(max(wake, 7 * 60), min(bed, 21 * 60)))
            en = st + timedelta(minutes=rng.uniform(1, 6))
            x.record(
                f"{HK_Q}WalkingSpeed",
                self.phone,
                st,
                en,
                rng.normal(4.9 + 0.3 * s.trend, 0.25) * (1 - 0.2 * s.illness),
                "km/hr",
            )
            x.record(f"{HK_Q}WalkingStepLength", self.phone, st, en, rng.normal(72 + 1.5 * s.trend, 2), "cm")
            x.record(f"{HK_Q}WalkingDoubleSupportPercentage", self.phone, st, en, rng.normal(0.285, 0.012), "%")
            if rng.random() < 0.6:
                x.record(f"{HK_Q}WalkingAsymmetryPercentage", self.phone, st, en, abs(rng.normal(0.0, 0.025)), "%")
        if rng.random() < 0.25:
            st = _at(d, rng.uniform(8 * 60, 20 * 60))
            x.record(
                f"{HK_Q}HeadphoneAudioExposure",
                self.phone,
                st,
                st + timedelta(minutes=rng.uniform(10, 50)),
                rng.normal(68, 5),
                "dBASPL",
            )
        if rng.random() < 0.3:  # the scale, a few mornings a week
            st = _at(d, wake + rng.uniform(5, 30))
            meta = {"HKWasUserEntered": "0", "HKTimeZone": tz_name}
            kg = s.weight + rng.normal(0, 0.25)
            x.record(f"{HK_Q}BodyMass", self.scale, st, st, kg, "kg", meta=meta)
            x.record(f"{HK_Q}BodyMassIndex", self.scale, st, st, kg / HEIGHT_M**2, "count", meta=meta)
            x.record(
                f"{HK_Q}BodyFatPercentage", self.scale, st, st, rng.normal(0.26 - 0.02 * s.trend, 0.006), "%", meta=meta
            )
        if s.i % 14 == 3:
            self._blood_pressure(s, _at(d, wake + rng.uniform(10, 40)))

    def _watch_day(self, s, watch, workouts, steps_h, work_min_h, work_kcal_h, awake, wake, bed, tz_name) -> None:
        rng, x, d = self.rng, self.x, s.day
        # Energy: active per waking hour, basal per 2 hours.
        active_h = np.where(awake, 4 + steps_h * 0.042, rng.uniform(0, 1.2, 24)) + work_kcal_h
        active_h *= rng.lognormal(0, 0.1, 24)
        for h in range(24):
            if active_h[h] >= 0.5:
                st = _at(d, h * 60 + rng.uniform(0, 3))
                x.record(
                    f"{HK_Q}ActiveEnergyBurned", watch, st, _at(d, h * 60 + rng.uniform(55, 59.9)), active_h[h], "kcal"
                )
        bmr = 10 * s.weight + 6.25 * HEIGHT_M * 100 - 5 * 36 - 161
        for h in range(0, 24, 2):
            x.record(
                f"{HK_Q}BasalEnergyBurned",
                watch,
                _at(d, h * 60),
                _at(d, h * 60 + 119.9),
                bmr / 12 * rng.normal(1, 0.02),
                "kcal",
            )
        # Exercise minutes, stand minutes and stand hours.
        brisk = rng.poisson(np.clip(steps_h / 1000 * 1.6, 0, 15) * awake)
        exercise_h = np.clip(brisk + np.round(work_min_h * 0.9), 0, 60)
        for h in np.nonzero(exercise_h >= 1)[0]:
            st = _at(d, h * 60 + rng.uniform(0, 5))
            x.record(
                f"{HK_Q}AppleExerciseTime",
                watch,
                st,
                st + timedelta(minutes=float(exercise_h[h])),
                exercise_h[h],
                "min",
            )
        stand_hours = 0
        for h in np.nonzero(awake)[0]:
            stood = rng.random() < (0.9 if steps_h[h] > 250 else 0.45)
            st = _at(d, h * 60)
            x.record(
                f"{HK_C}AppleStandHour",
                watch,
                st,
                _at(d, h * 60 + 59.99),
                "HKCategoryValueAppleStandHourStood" if stood else "HKCategoryValueAppleStandHourIdle",
            )
            if stood:
                stand_hours += 1
                m = float(rng.integers(2, 13))
                x.record(f"{HK_Q}AppleStandTime", watch, _at(d, h * 60 + 5), _at(d, h * 60 + 5 + m), m, "min")
        # Heart rate: every ~15 min awake, denser during workouts.
        in_workout = np.zeros(24 * 60, bool)
        for w in workouts:
            in_workout[int(w.start) : int(min(w.start + w.minutes, 24 * 60))] = True
        t = max(wake, 0) + rng.uniform(0, 10)
        day_end = min(bed, 24 * 60 - 1)
        while t < day_end:
            if not in_workout[int(t)]:
                h = int(t // 60)
                busy = steps_h[h] > 400
                hr = s.rhr + 12 + steps_h[h] / 90 + rng.normal(0, 4) + 9 * s.fever
                x.record(
                    f"{HK_Q}HeartRate",
                    watch,
                    _at(d, t),
                    _at(d, t),
                    round(hr),
                    "count/min",
                    meta={"HKMetadataKeyHeartRateMotionContext": "2" if busy else "1"},
                    created=_at(d, t + rng.uniform(1, 4)),
                )
            t += rng.uniform(11, 19)
        for w in workouts:
            self._workout(s, watch, w, tz_name)
        # Resting / walking HR, HRV (with beat-to-beat list), audio, daylight.
        x.record(f"{HK_Q}RestingHeartRate", watch, _at(d, 5), _at(d, 22 * 60 + 40), round(s.rhr), "count/min")
        if s.steps > 1500:
            x.record(
                f"{HK_Q}WalkingHeartRateAverage",
                watch,
                _at(d, 6 * 60),
                _at(d, 22 * 60 + 40),
                round(s.rhr + 36 + rng.normal(0, 3) - 3 * s.trend),
                "count/min",
            )
        for _ in range(int(rng.integers(2, 4))):
            self._hrv(s, watch, rng.uniform(max(wake, 0) + 15, day_end - 5))
        for _ in range(int(rng.integers(4, 9))):
            st = _at(d, rng.uniform(max(wake, 0), day_end - 30))
            x.record(
                f"{HK_Q}EnvironmentalAudioExposure",
                watch,
                st,
                st + timedelta(minutes=30),
                rng.normal(63 + (4 if s.weekday < 5 else 0), 6),
                "dBASPL",
            )
        daylight = max(
            0.0,
            30
            + 45 * s.season
            + 25 * (s.weekday >= 5)
            + sum(w.minutes for w in workouts if not w.indoor) * 0.8
            + rng.normal(0, 15),
        ) * (1 - 0.7 * s.illness)
        n_dl = int(rng.integers(1, 4))
        for k in range(n_dl):
            st = _at(d, 9 * 60 + k * 180 + rng.uniform(0, 90))
            x.record(
                f"{HK_Q}TimeInDaylight", watch, st, st + timedelta(minutes=daylight / n_dl), daylight / n_dl, "min"
            )
        if rng.random() < 0.4:
            st = _at(d, rng.uniform(9 * 60, 19 * 60))
            x.record(
                f"{HK_C}HandwashingEvent",
                watch,
                st,
                st + timedelta(seconds=float(rng.uniform(18, 40))),
                "HKCategoryValueNotApplicable",
            )
        if rng.random() < 0.12:
            st = _at(d, rng.uniform(7 * 60, 22 * 60))
            x.record(
                f"{HK_C}MindfulSession",
                watch,
                st,
                st + timedelta(minutes=float(rng.integers(1, 11))),
                "HKCategoryValueNotApplicable",
            )
        goal = 450 if s.i < self.n // 2 else 550
        self._summary(d, float(active_h.sum()), int(exercise_h.sum()), stand_hours, goal)

    def _summary(self, d: date, kcal: float, exercise: int, stand: int, goal: int = 450) -> None:
        self.activity_summaries.append(
            f' <ActivitySummary dateComponents="{d:%Y-%m-%d}" activeEnergyBurned="{_num(kcal, 3)}" '
            f'activeEnergyBurnedGoal="{goal}" activeEnergyBurnedUnit="kcal" appleMoveTime="0" appleMoveTimeGoal="0" '
            f'appleExerciseTime="{exercise}" appleExerciseTimeGoal="30" appleStandHours="{stand}" appleStandHoursGoal="12"/>'
        )

    def _hrv(self, s: _Day, watch: Source, t: float) -> None:
        rng, d = self.rng, s.day
        sdnn = s.hrv * rng.lognormal(0, 0.18)
        start = _at(d, t)
        mean_rr = 60_000 / (s.rhr + rng.uniform(3, 12))
        beats, clock = [], 0.0
        wall = self.clock.utc(start) + self.clock.export_offset
        for _ in range(int(rng.integers(35, 55))):
            rr = max(350.0, rng.normal(mean_rr, sdnn))
            clock += rr / 1000
            ts = wall + timedelta(seconds=clock)
            frac = f"{ts.microsecond // 10_000:02d}"
            stamp = f"{ts:%H:%M:%S},{frac}" if self.german else f"{ts.hour % 12 or 12}:{ts:%M:%S}.{frac} {ts:%p}"
            beats.append(f'   <InstantaneousBeatsPerMinute bpm="{round(60_000 / rr)}" time="{stamp}"/>')
        children = ["  <HeartRateVariabilityMetadataList>", *beats, "  </HeartRateVariabilityMetadataList>"]
        self.x.record(
            f"{HK_Q}HeartRateVariabilitySDNN",
            watch,
            start,
            start + timedelta(seconds=clock),
            sdnn,
            "ms",
            children=children,
        )

    def _night(self, s: _Day, night: _Night, nxt: _Day) -> None:
        rng, x, d = self.rng, self.x, s.day
        watch = self.watch(d)
        x.record(
            f"{HK_C}SleepAnalysis",
            self.phone_plain,
            _at(d, night.bed - rng.uniform(5, 20)),
            _at(d, night.wake + 3),
            SLEEP["in_bed"],
        )
        if not nxt.watch:  # watch off overnight: only the iPhone's in-bed estimate
            return
        for stage, a, b in night.stages:
            x.record(f"{HK_C}SleepAnalysis", watch, _at(d, a), _at(d, b), SLEEP[stage], created=_at(d, night.wake + 5))
        # Sleeping heart rate, SpO2, respiratory rate and (late in the period) wrist temperature.
        rhr = nxt.rhr
        t = night.bed + rng.uniform(5, 25)
        while t < night.wake:
            x.record(
                f"{HK_Q}HeartRate",
                watch,
                _at(d, t),
                _at(d, t),
                round(rhr - 3 + rng.normal(0, 2) + 4 * nxt.fever),
                "count/min",
                meta={"HKMetadataKeyHeartRateMotionContext": "1"},
            )
            t += rng.uniform(30, 50)
        for _ in range(int(rng.integers(3, 7))):
            st = _at(d, rng.uniform(night.bed + 20, night.wake - 10))
            x.record(
                f"{HK_Q}OxygenSaturation", watch, st, st, float(np.clip(rng.normal(nxt.spo2, 0.008), 0.9, 1.0)), "%"
            )
        for _ in range(int(rng.integers(3, 7))):
            st = _at(d, rng.uniform(night.bed + 20, night.wake - 10))
            x.record(
                f"{HK_Q}RespiratoryRate",
                watch,
                st,
                st + timedelta(minutes=float(rng.uniform(5, 30))),
                rng.normal(nxt.resp, 0.5),
                "count/min",
            )
        if s.i > self.n * 0.6:
            x.record(
                f"{HK_Q}AppleSleepingWristTemperature",
                watch,
                _at(d, night.bed + 30),
                _at(d, night.wake - 15),
                rng.normal(35.3 + nxt.fever, 0.15),
                "degC",
            )

    def _blood_pressure(self, s: _Day, t: datetime) -> None:
        rng = self.rng
        sys_ = round(rng.normal(117 - 3 * s.trend + 6 * s.illness, 5))
        dia = round(rng.normal(76 - 2 * s.trend, 4))
        meta = {"HKWasUserEntered": "0"}
        recs = [
            self.x.record(f"{HK_Q}BloodPressureSystolic", self.cuff, t, t, sys_, "mmHg", meta=meta),
            self.x.record(f"{HK_Q}BloodPressureDiastolic", self.cuff, t, t, dia, "mmHg", meta=meta),
        ]
        c = self.clock
        self.x.correlations.append(
            "\n".join(
                [
                    f' <Correlation type="HKCorrelationTypeIdentifierBloodPressure" {self.cuff.attrs()} '
                    f'creationDate="{c.stamp(t)}" startDate="{c.stamp(t)}" endDate="{c.stamp(t)}">',
                    '  <MetadataEntry key="HKWasUserEntered" value="0"/>',
                    *[" " + r.replace("\n", "\n ") for r in recs],
                    " </Correlation>",
                ]
            )
        )

    def _workout(self, s: _Day, watch: Source, w: _Workout, tz_name: str) -> None:
        rng, x, c, d = self.rng, self.x, self.clock, s.day
        k = w.kind
        start, end = _at(d, w.start), _at(d, w.start + w.minutes)
        # Heart rate every ~2 minutes with warm-up ramp.
        t = 0.5
        while t < w.minutes:
            ramp = min(1.0, t / 8)
            hr = w.hr_min + (w.hr_avg - w.hr_min) * ramp * 1.08 + rng.normal(0, 4)
            x.record(
                f"{HK_Q}HeartRate",
                watch,
                _at(d, w.start + t),
                _at(d, w.start + t),
                round(float(np.clip(hr, w.hr_min, w.hr_max))),
                "count/min",
                meta={"HKMetadataKeyHeartRateMotionContext": "2"},
            )
            t += rng.uniform(1.5, 2.5)
        if k.distance_type in ("DistanceCycling", "DistanceSwimming"):
            unit, val = ("m", w.distance_km * 1000) if k.distance_type == "DistanceSwimming" else ("km", w.distance_km)
            x.record(f"{HK_Q}{k.distance_type}", watch, start, end, val, unit)
        if k.hk == "Swimming":
            x.record(f"{HK_Q}SwimmingStrokeCount", watch, start, end, round(w.distance_km * 1000 / 25 * 18), "count")
        if not w.indoor and k.distance_type == "DistanceWalkingRunning" and rng.random() < 0.35:
            x.record(
                f"{HK_Q}VO2Max",
                watch,
                end,
                end,
                rng.normal(36 + 7 * s.trend - 0.3 * (s.rhr - 58), 0.8),
                "mL/min·kg",
                meta={"HKVO2MaxTestType": "2"},
            )

        meta = {"HKIndoorWorkout": "1" if w.indoor else "0", "HKTimeZone": tz_name}
        if not w.indoor:
            meta["HKWeatherTemperature"] = f"{round(w.temp_c * 9 / 5 + 32)} degF"
            meta["HKWeatherHumidity"] = f"{round(w.humidity * 100)} %"
        if k.hk == "Swimming":
            meta |= {"HKSwimmingLocationType": "1", "HKLapLength": "25 m"}
        meta["HKAverageMETs"] = f"{_num(w.active_kcal / w.minutes * 60 / s.weight + 1, 5)} kcal/hr·kg"
        if w.elevation_m:
            meta["HKElevationAscended"] = f"{round(w.elevation_m * 100)} cm"
        # Older exports put distance/energy on the Workout itself, newer only in WorkoutStatistics.
        totals = ""
        if s.i < self.n // 3:
            totals = f' totalEnergyBurned="{_num(w.active_kcal)}" totalEnergyBurnedUnit="kcal"'
            if w.distance_km:
                dist, unit = (w.distance_km * 1000, "m") if k.hk == "Swimming" else (w.distance_km, "km")
                totals = f' totalDistance="{_num(dist)}" totalDistanceUnit="{unit}"' + totals
        lines = [
            f' <Workout workoutActivityType="HKWorkoutActivityType{k.hk}" duration="{_num(w.minutes)}" '
            f'durationUnit="min"{totals} {watch.attrs()} creationDate="{c.stamp(end + timedelta(minutes=1))}" '
            f'startDate="{c.stamp(start)}" endDate="{c.stamp(end)}">'
        ]
        lines += [f'  <MetadataEntry key="{key}" value="{v}"/>' for key, v in meta.items()]
        if w.minutes > 20 and rng.random() < 0.35:
            p = w.minutes * rng.uniform(0.3, 0.7)
            lines.append(f'  <WorkoutEvent type="HKWorkoutEventTypePause" date="{c.stamp(_at(d, w.start + p))}"/>')
            lines.append(
                f'  <WorkoutEvent type="HKWorkoutEventTypeResume" date="{c.stamp(_at(d, w.start + p + rng.uniform(0.5, 4)))}"/>'
            )
        if k.hk in ("Running", "Walking", "Hiking") and w.distance_km >= 1:
            per_km = w.minutes / w.distance_km
            for km in range(int(w.distance_km)):
                lines.append(
                    f'  <WorkoutEvent type="HKWorkoutEventTypeSegment" date="{c.stamp(_at(d, w.start + km * per_km))}" '
                    f'duration="{_num(per_km * rng.normal(1, 0.03))}" durationUnit="min"/>'
                )
        if k.hk == "Swimming":
            for lap in range(0, int(w.distance_km * 1000 / 25), 4):
                tl = w.start + w.minutes * lap * 25 / max(w.distance_km * 1000, 1)
                lines.append(
                    f'  <WorkoutEvent type="HKWorkoutEventTypeLap" date="{c.stamp(_at(d, tl))}" '
                    f'duration="{_num(w.minutes * 25 / (w.distance_km * 1000))}" durationUnit="min"/>'
                )
        se = f'startDate="{c.stamp(start)}" endDate="{c.stamp(end)}"'
        lines.append(
            f'  <WorkoutStatistics type="{HK_Q}ActiveEnergyBurned" {se} sum="{_num(w.active_kcal)}" unit="kcal"/>'
        )
        lines.append(
            f'  <WorkoutStatistics type="{HK_Q}BasalEnergyBurned" {se} sum="{_num(w.basal_kcal)}" unit="kcal"/>'
        )
        if w.distance_km:
            dist, unit = (w.distance_km * 1000, "m") if k.hk == "Swimming" else (w.distance_km, "km")
            lines.append(f'  <WorkoutStatistics type="{HK_Q}{k.distance_type}" {se} sum="{_num(dist)}" unit="{unit}"/>')
        lines.append(
            f'  <WorkoutStatistics type="{HK_Q}HeartRate" {se} average="{_num(w.hr_avg, 3)}" '
            f'minimum="{round(w.hr_min)}" maximum="{round(w.hr_max)}" unit="count/min"/>'
        )
        if k.hk == "Swimming":
            lines.append(
                f'  <WorkoutStatistics type="{HK_Q}SwimmingStrokeCount" {se} sum="{round(w.distance_km * 720)}" unit="count"/>'
            )
        if not w.indoor and k.outdoor and w.distance_km > 0.3:
            name = _route_name(start)
            while name in self.routes:
                name = name.replace(".gpx", "_1.gpx")
            w.route = name
            self.routes[name] = _gpx(rng, w, c.utc(start).replace(tzinfo=UTC))
            lines += [
                f'  <WorkoutRoute {watch.attrs()} creationDate="{c.stamp(end + timedelta(minutes=2))}" {se}>',
                '   <MetadataEntry key="HKMetadataKeySyncVersion" value="2"/>',
                f'   <MetadataEntry key="HKMetadataKeySyncIdentifier" value="{_uuid(rng)}"/>',
                f'   <FileReference path="/workout-routes/{name}"/>',
                "  </WorkoutRoute>",
            ]
        lines.append(" </Workout>")
        x.workouts.append("\n".join(lines))
        x.counts["Workout"] = x.counts.get("Workout", 0) + 1

    # ----------------------------------------------------------------- side files
    def _ecgs(self, states: list[_Day]) -> None:
        plan = [
            ("Sinus Rhythm", "sinus", ""),
            ("Sinus Rhythm", "sinus", ""),
            ("Sinus Rhythm", "sinus", "Fatigue"),
            ("Atrial Fibrillation", "afib", ""),
            ("Sinus Rhythm", "sinus", ""),
            ("Inconclusive", "sinus", ""),
        ]
        idx = np.linspace(self.n * 0.05, self.n * 0.95, len(plan)).astype(int)
        for k, ((label, rhythm, symptoms), i) in enumerate(zip(plan, idx, strict=True)):
            s = states[int(i)]
            local = _at(s.day, self.rng.uniform(11 * 60, 20 * 60)).replace(microsecond=0)
            noise = 70.0 if label == "Inconclusive" else 12.0
            hr = s.rhr + self.rng.uniform(4, 14) + (25 if rhythm == "afib" else 0)
            sig = ecg_signal(self.rng, hr=hr, rhythm=rhythm, noise_uv=noise)
            name = f"ecg_{s.day:%Y-%m-%d}.csv"
            while name in self.ecg_files:
                name = name.replace(".csv", "_1.csv")
            self.ecg_files[name] = _ecg_csv(sig, self.clock.true_stamp(local), label, symptoms, german=k % 2 == 0)

    def _clinical(self) -> None:
        items = [
            (
                "Immunization",
                self.start + timedelta(days=int(self.n * 0.3)),
                "Influenza, seasonal, injectable",
                "Influenza",
                "Example Vaccines Ltd.",
                1,
            ),
            (
                "Immunization",
                self.start + timedelta(days=int(self.n * 0.8)),
                "Tetanus and diphtheria toxoids",
                "Tetanus",
                "Example Biologicals",
                3,
            ),
        ]
        received = self.clock.stamp(self.export_local - timedelta(days=2))
        for rtype, day, vaccine, disease, maker, dose in items:
            rid = _uuid(self.rng)
            res = {
                "resourceType": "Immunization",
                "id": rid.lower(),
                "status": "completed",
                "vaccineCode": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/cvx",
                            "code": "141" if disease == "Influenza" else "139",
                            "display": vaccine,
                        }
                    ],
                    "text": vaccine,
                },
                "patient": {"display": PERSON},
                "occurrenceDateTime": f"{day:%Y-%m-%d}T10:30:00Z",
                "primarySource": True,
                "manufacturer": {"display": maker},
                "lotNumber": f"SYN{int(self.rng.integers(10000, 99999))}",
                "protocolApplied": [
                    {
                        "targetDisease": [{"coding": [{"display": disease}]}],
                        "targetedDisease": {"coding": [{"display": disease}]},
                        "doseNumberPositiveInt": dose,
                    }
                ],
            }
            self._add_clinical(rtype, rid, res, "HKClinicalTypeIdentifierImmunizationRecord", received)
        rid = _uuid(self.rng)
        day = self.start + timedelta(days=int(self.n * 0.5))
        obs = {
            "resourceType": "Observation",
            "id": rid.lower(),
            "status": "final",
            "category": [
                {
                    "coding": [
                        {"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "2093-3",
                        "display": "Cholesterol [Mass/volume] in Serum or Plasma",
                    }
                ]
            },
            "subject": {"display": PERSON},
            "effectiveDateTime": f"{day:%Y-%m-%d}T08:15:00Z",
            "valueQuantity": {
                "value": round(float(self.rng.normal(182, 12)), 1),
                "unit": "mg/dL",
                "system": "http://unitsofmeasure.org",
                "code": "mg/dL",
            },
        }
        self._add_clinical("Observation", rid, obs, "HKClinicalTypeIdentifierLabResultRecord", received)

    def _add_clinical(self, rtype: str, rid: str, res: dict, hk_type: str, received: str) -> None:
        fname = f"{rtype}-{rid}.json"
        self.clinical[fname] = json.dumps(res, indent=2)
        self.clinical_xml.append(
            f' <ClinicalRecord type="{hk_type}" identifier="{rid.lower()}" sourceName="Example Health Clinic" '
            f'sourceURL="https://fhir.example.org/r4/{rtype}/{rid.lower()}" fhirVersion="4.0.1" '
            f'receivedDate="{received}" resourceFilePath="/clinical-records/{fname}"/>'
        )

    # ----------------------------------------------------------------- write
    def write(self) -> dict[str, int]:
        out = self.out
        out.mkdir(parents=True, exist_ok=True)
        for sub in ("workout-routes", "electrocardiograms", "clinical-records"):
            (out / sub).mkdir(exist_ok=True)
            for old in (out / sub).iterdir():
                if old.is_file():
                    old.unlink()
        me = (
            'HKCharacteristicTypeIdentifierDateOfBirth="1990-04-01" '
            'HKCharacteristicTypeIdentifierBiologicalSex="HKBiologicalSexFemale" '
            'HKCharacteristicTypeIdentifierBloodType="HKBloodTypeNotSet" '
            'HKCharacteristicTypeIdentifierFitzpatrickSkinType="HKFitzpatrickSkinTypeNotSet" '
            'HKCharacteristicTypeIdentifierCardioFitnessMedicationsUse="None"'
        )
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            _DTD,
            f'<HealthData locale="{self.locale}">',
            f' <ExportDate value="{self.clock.stamp(self.export_local)}"/>',
            f" <Me {me}/>",
            *self.x.xml_records(),
            *self.x.correlations,
            *self.x.workouts,
            *self.activity_summaries,
            *self.clinical_xml,
            "</HealthData>",
            "",
        ]
        (out / "export.xml").write_text("\n".join(parts), encoding="utf-8")
        (out / "export_cda.xml").write_text(_cda(self.clock.stamp(self.export_local)), encoding="utf-8")
        for name, text in self.routes.items():
            (out / "workout-routes" / name).write_text(text, encoding="utf-8")
        for name, text in self.ecg_files.items():
            (out / "electrocardiograms" / name).write_text(text, encoding="utf-8")
        for name, text in self.clinical.items():
            (out / "clinical-records" / name).write_text(text, encoding="utf-8")
        counts = dict(sorted(self.x.counts.items()))
        counts |= {
            "Correlation": len(self.x.correlations),
            "ActivitySummary": len(self.activity_summaries),
            "routes": len(self.routes),
            "ecgs": len(self.ecg_files),
            "clinical_records": len(self.clinical),
        }
        return counts


def _cda(export_stamp: str) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<ClinicalDocument xmlns="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
            ' <realmCode code="US"/>',
            ' <typeId root="2.16.840.1.113883.1.3" extension="POCD_HD000040"/>',
            " <title>Health Export CDA</title>",
            f' <effectiveTime value="{export_stamp[:19].replace("-", "").replace(":", "").replace(" ", "")}"/>',
            " <recordTarget><patientRole><patient><name><given>Jane</given><family>Appleseed</family></name>"
            "</patient></patientRole></recordTarget>",
            " <component><structuredBody/></component>",
            "</ClinicalDocument>",
            "",
        ]
    )


def generate(
    out_dir: str | Path, days: int = 730, seed: int = 42, tz: str = "Europe/Berlin", end: date = DEFAULT_END
) -> dict[str, int]:
    """Write a synthetic Apple Health export folder to `out_dir`; returns element counts per type."""
    if days < 7:
        raise ValueError("days must be >= 7")
    ZoneInfo(tz)  # fail early on a bad zone name
    t0 = time.time()
    g = _Generator(Path(out_dir), days, seed, tz, end)
    g.run()
    counts = g.write()
    size = sum(p.stat().st_size for p in Path(out_dir).rglob("*") if p.is_file())
    print(
        f"synthetic export: {days} days ({g.start}..{g.end}, {tz}, seed {seed}) -> {out_dir} "
        f"[{size / 1e6:.1f} MB, {time.time() - t0:.1f}s]",
        flush=True,
    )
    return counts
