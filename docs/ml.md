# Machine Learning on Your Health Data

The project ships a PyTorch suite (`ml/healthml`, run with `ml/run_all.py`) with seven experiments.
Each one answers a concrete question about your own data. This page explains, for each experiment,
which methods it uses, what it can detect and where it reaches its limits.

> Everything here is exploratory analysis of **one person's** data. It is not medical advice. Flags
> and scores are signals to look at, not a diagnosis.

## Run It

```bash
make ml                                   # = .venv/bin/python ml/run_all.py (a few minutes on CPU or Apple MPS)
.venv/bin/python ml/run_all.py --quick    # 1 seed, skips the pretrained Chronos model
make demo                                 # the same on a synthetic export, no personal data needed
```

Outputs:

- `data/ml/*.parquet`, shown on the app's **ML insights** page;
- `ml/reports/metrics.md` and `ml/reports/figures/` with all scores and charts (git-ignored, because
  they describe you);
- on Databricks, the `ml_*` Delta tables, plus MLflow runs under `/Shared/health_ml`.

## The Seven Experiments at a Glance

| # | Question | Main method | Compared against | Shown in |
|---|---|---|---|---|
| 1 | What will my resting heart rate and HRV be tomorrow? | Linear regression, MLP, GRU/LSTM, Transformer (PyTorch); Chronos-Bolt (pretrained) | Persistence, 7-day mean, same weekday last week, gradient boosting | App: *Forecasts* |
| 2 | Will I close my move ring / work out tomorrow? | Logistic regression (PyTorch) | Base rate, Naive Bayes, decision tree, random forest, gradient boosting, SVM, KNN | Report |
| 3 | Which days were unusual, or looked like I was getting ill? | Masked autoencoder (PyTorch); rule-based illness score | PCA reconstruction | App: *Unusual days* |
| 4 | Which days resemble this one? | K-nearest neighbours in the autoencoder's embedding | – | App: *Similar days* |
| 5 | What kinds of days and workouts do I have? | PCA + k-means (PyTorch) | Ward hierarchical clustering, DBSCAN | App: *Day types*, *Workout clusters* |
| 6 | Do any heartbeats in my ECGs look odd? | 1D convolutional autoencoder on single beats (PyTorch); RR-interval rule | PCA reconstruction | App: *ECG beats* |
| 7 | How hard will a workout be? | Random forest, MLP (PyTorch) | Median per activity, ridge, decision tree, gradient boosting, KNN, SVR | Report |

A descriptive part (`insights`) adds trends, weekday and seasonal effects, and how workouts relate to
next-day HRV. It uses plain statistics, not models.

## 1. Forecasting Resting Heart Rate and HRV

**How it works.** Every day becomes a row of features: recent heart values (lags), 7- and 28-day
rolling means, activity and training load, calendar. The models predict tomorrow's deviation from the 7-day mean,
which keeps them stable when your fitness changes over the years. The neural sequence models
(GRU, LSTM, a small Transformer) read 28-day windows instead of rows. Chronos-Bolt-small, a
pretrained time-series model (~190 MB from Hugging Face), forecasts without any training.

**What it can detect.**

- A realistic expectation for tomorrow, and how far today's value deviates from it.
- Which features drive the forecast: the linear coefficients and the permutation importances show,
  for example, whether a hard workout or a high training load moves your resting HR the next day.
- How predictable your values are at all: the *skill vs persistence* column shows how much better
  than "tomorrow = today" a model is.

**What to expect.** Simple linear regression usually ties with the neural models on a few thousand
noisy daily rows. Day-level values are noisy, so weekly averages are more useful for decisions.

## 2. Predicting Tomorrow's Behaviour

**How it works.** Logistic regression predicts two yes/no labels for the next day: *move ring
closed* and *workout*. It uses the same kind of features, especially your habits over the last 28
days. Six classic classifiers run on the same split as a comparison, and every model is checked for
calibration (does "70%" really happen 70% of the time?).

**What it can detect.**

- How strongly your behaviour follows routine: weekday, recent streaks, ring-closure rate.
- Days with a low predicted chance of closing your ring, i.e. where a nudge would matter.

**What to expect.** Habits dominate. Logistic regression is as good as the more complex models and
easier to explain. Naive Bayes is included as a negative example: the features are correlated, so
its probabilities are overconfident.

## 3. Unusual Days and Illness-Like Days

**How it works.** Each day is a vector of physiology and activity: resting HR, HRV, average and
minimum HR, walking HR, SpO₂, respiratory rate, steps, active energy, exercise minutes, plus resting
HR, HRV and average HR relative to your own 60-day baseline. A masked autoencoder learns to
compress and rebuild typical days. Missing values are masked out, so gaps don't count as anomalies.
Days it rebuilds badly get a high **anomaly score**; the top 2% are flagged. Consecutive flagged days
are merged into episodes.

A second, rule-based **illness score** measures "resting HR above and HRV below your baseline", both
in standard deviations.

**What it can detect.**

- **Unusual days** (`anomaly_score`): travel, a sensor glitch, an exceptionally hard training day,
  a day without the watch, or an odd combination of values. Each flagged day lists the features
  that contributed most.
- **Illness-like days** (`illness_score`): the pattern that often comes with a cold, a fever,
  overtraining, alcohol or very poor sleep. On the synthetic demo data it finds 9 of 10 hidden
  illness days.

**What to expect.** The autoencoder learns that "resting HR up, HRV down" usually go together, so it
rebuilds illness-like days quite well and misses them. Use it for "something was different" and the
illness score for "my body was under strain". The PCA score is shown next to the autoencoder as a
simple linear reference.

## 4. Similar Days

**How it works.** The autoencoder compresses each day into a few numbers (an embedding). For any day,
k-nearest neighbours finds the five closest days in that space, ignoring the days right around it.

**What it can detect.** Earlier days that looked like a given one, for example "the last times my
values looked like today". This helps to recognise recurring situations such as travel, illness or
race weeks.

## 5. Day Types and Workout Types

**How it works.** Days are described by behaviour (steps, walking and cycling distance, flights
climbed, active energy, exercise, standing and workout minutes). Workouts are described by duration,
energy per minute, heart rate, speed, whether they had a distance, start hour and indoor/outdoor. PCA projects
them onto a 2-D map. K-means groups them, with the number of groups chosen by silhouette score. Each
group is named from what makes it stand out, e.g. *rest day*, *high-step day*, *long workout day*.
Ward clustering and DBSCAN run as cross-checks.

**What it can detect.**

- Your typical kinds of days and how their share changes over months and years.
- Distinct workout styles within one activity, e.g. easy vs. hard runs or commute vs. long rides.
- Outlier workouts, which DBSCAN marks as noise.

## 6. ECG Beat Anomalies

**How it works.** Each Apple Watch ECG recording is filtered, the R-peaks are detected, and every
heartbeat is cut out and normalised. A 1D convolutional autoencoder learns the shape of your normal
beats from earlier "Sinus Rhythm" recordings. A beat is flagged for its **shape** (rebuilt badly) or
its **rhythm** (the interval before it is more than 25% off the recording's median).

**What it can detect.**

- Single beats with an unusual shape, often motion artefacts or noise, sometimes extra beats.
- Irregular beat-to-beat intervals, as in recordings classified as atrial fibrillation.
- Heart rate and beat-to-beat variability per recording.

**What to expect.** It can't diagnose anything: there are too few abnormal recordings per person to
train a classifier, and Apple's own classification stays the reference.

## 7. Workout Effort

**How it works.** Regression models predict a workout's active energy (kcal) and average heart rate
from what you know before you start: activity, duration, distance, speed, elevation, weather, time of
day. Heart rate and METs are deliberately left out as inputs, because Apple calculates the kcal from
them.

**What it can detect.**

- Which factors make a workout costly for you (permutation importances): duration, elevation,
  temperature, activity type.
- Workouts that were much harder or easier than expected for their conditions.

**What to expect.** A random forest works best on this small table; ridge regression is close.

## How Much Data You Need

| Experiment | Needs | Source table |
|---|---|---|
| Forecasts | ≥ ~1 year of daily resting HR / HRV from an Apple Watch | `gold_daily_summary` |
| Behaviour prediction | ≥ ~6 months of activity rings | `gold_daily_summary` |
| Unusual days, similar days | ≥ ~6 months of resting HR, HRV, steps | `gold_daily_summary` |
| Day types | ≥ ~3 months | `gold_daily_summary` |
| Workout types, workout effort | ≥ ~100 workouts | `gold_workouts` |
| ECG beats | ≥ 3 ECG recordings | `silver_ecg_samples` |

Experiments without enough data are skipped with a message, so `run_all.py` always finishes.

## Guardrails Built In

- **No look into the future.** Features for day `t` only use days up to `t`; scalers are fit on the
  training rows only. `tests/test_ml.py` changes every value after `t` and checks that the features
  stay the same.
- **Splits by time, relative to your data.** The most recent ~20% of days are test and the ~15%
  before them are validation (used for early stopping). `HEALTHML_VAL_START` / `HEALTHML_TEST_START`
  pin fixed dates instead. Nothing is shuffled across time.
- **Start where the sensor is reliable.** Each experiment starts on the first day its key column is
  present on at least 80% of days in a 90-day window.
- **Baselines first.** Every forecast is compared with simple rules, every classifier with the base
  rate, every neural anomaly score with PCA.
- **Associations, not causes.** Effects are measured within your own history and don't prove that
  one thing causes another.

## Not Modelled, and Why

- **Sleep:** many people don't wear the watch at night, so sleep data is often sparse. It is in the
  gold layer and in the app, but not used for ML.
- **GPS routes:** route points are in silver, if you want to try segment or speed models.

## Code Map

```
ml/
  healthml/
    config.py            paths (HEALTH_DATA_DIR), seeds, device (MPS > CUDA > CPU; HEALTHML_DEVICE overrides), time split
    sources.py           ParquetSource (local) and SparkSource (Unity Catalog tables)
    data.py              leakage-safe feature building, standardizer
    models.py            linear/logistic, MLP, GRU/LSTM, Transformer, masked AE, 1D conv AE, torch k-means, PCA
    train.py             training loop with early stopping and refit mode
    evaluation.py        MAE/RMSE/R², AUC, Brier, log loss, calibration
    ecg.py               band-pass filter, R-peak detection, beat segmentation, HRV
    tracking.py          optional MLflow logging (no-op locally)
    experiments/         forecast, classify, anomaly (+ similar days), clustering, ecg_cnn, workouts, insights
  run_all.py             runs everything → data/ml/*.parquet, ml/reports/
databricks/notebooks/ml/ 10–14: the same experiments on Databricks, writing ml_* tables and MLflow runs
tests/test_ml.py         leakage, shapes, k-means, PCA, R-peak tests
```
