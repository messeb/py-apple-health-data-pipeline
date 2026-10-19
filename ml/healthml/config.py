"""Paths, seeds, device selection and the time splits shared by all experiments.

Paths follow the environment like the Streamlit app does:

* ``HEALTH_DATA_DIR``     lakehouse root with ``silver/``, ``gold/``; outputs go to ``<dir>/ml`` (default ``data/``)
* ``HEALTHML_REPORT_DIR`` metrics and figures (default ``ml/reports``)
* ``HEALTHML_VAL_START`` / ``HEALTHML_TEST_START`` pin the split to fixed dates. Default: relative to
  the data (last ~20% of days test, ~15% before it validation)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("HEALTH_DATA_DIR", REPO_ROOT / "data")).expanduser().resolve()
OUT_DIR = DATA_DIR / "ml"
REPORT_DIR = Path(os.environ.get("HEALTHML_REPORT_DIR", REPO_ROOT / "ml" / "reports")).expanduser().resolve()
FIG_DIR = REPORT_DIR / "figures"
HF_CACHE = Path(os.environ.get("HF_HOME", REPO_ROOT / "hf_cache"))

SEED = 42


class NotEnoughData(ValueError):
    """Raised when an experiment has too little history to train/validate/test; callers skip it."""


@dataclass(frozen=True)
class TimeSplit:
    """Chronological split: train < val_start <= val < test_start <= test.

    Without explicit dates the boundaries are relative to the data being split: the last
    ``test_frac`` of the distinct dates is test and the ``val_frac`` before it is validation.
    """

    val_start: str | None = os.environ.get("HEALTHML_VAL_START")
    test_start: str | None = os.environ.get("HEALTHML_TEST_START")
    val_frac: float = 0.15
    test_frac: float = 0.20
    min_train: int = 60
    min_eval: int = 20

    def resolve(self, dates) -> TimeSplit:
        """Fix the boundaries for ``dates`` (explicit dates win) and check every part is big enough."""
        import pandas as pd

        d = np.sort(pd.to_datetime(pd.Series(dates)).dropna().unique())
        if len(d) == 0:
            raise NotEnoughData("no dates to split")
        val_start, test_start = self.val_start, self.test_start
        if test_start is None:
            test_start = str(pd.Timestamp(d[int(len(d) * (1 - self.test_frac))]).date())
        if val_start is None:
            before = d[d < np.datetime64(test_start)]
            if len(before) == 0:
                raise NotEnoughData("no dates before the test period")
            k = int(len(d) * self.val_frac)
            val_start = str(pd.Timestamp(before[max(0, len(before) - k)]).date())
        out = replace(self, val_start=val_start, test_start=test_start)
        parts = out._label(d)
        n = {p: int((parts == p).sum()) for p in ("train", "val", "test")}
        if n["train"] < self.min_train or n["val"] < self.min_eval or n["test"] < self.min_eval:
            raise NotEnoughData(f"split {val_start}/{test_start} leaves too few days: {n}")
        return out

    def _label(self, d) -> np.ndarray:
        out = np.full(len(d), "train", dtype=object)
        out[d >= np.datetime64(self.val_start)] = "val"
        out[d >= np.datetime64(self.test_start)] = "test"
        return out

    def label(self, dates) -> np.ndarray:
        import pandas as pd

        split = self if self.val_start and self.test_start else self.resolve(dates)
        return split._label(pd.to_datetime(pd.Series(dates)).to_numpy())


DEFAULT_SPLIT = TimeSplit()


def first_watch_day(daily, column: str = "resting_hr", coverage: float = 0.8) -> str | None:
    """First day from which ``column`` is reliably recorded: the start of the first 90-day
    window with at least ``coverage`` of days present (2018-01 for this export's resting HR;
    2017 has only ~11% coverage). Falls back to the first non-null day for short exports."""
    if column not in daily:
        return None
    present = daily[column].notna().astype(float)
    if not present.any():
        return None
    window = min(90, max(7, len(present) // 4))
    rate = present.rolling(window, min_periods=window).mean()
    ok = rate[rate >= coverage]
    if ok.empty:
        return str(daily[column].dropna().index[0].date())
    start = ok.index[0] - pd.Timedelta(days=window - 1)
    return str(daily[column].loc[start:].dropna().index[0].date())


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """MPS on Apple silicon, CUDA on a GPU cluster, else CPU. HEALTHML_DEVICE overrides."""
    forced = os.environ.get("HEALTHML_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs or (OUT_DIR, REPORT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
