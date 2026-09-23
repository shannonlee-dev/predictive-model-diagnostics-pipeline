"""Validate observations and prepare strictly causal, train-scaled windows."""

import numpy as np
import pandas as pd

TRAIN_END_FRACTION = 0.7
VALIDATION_END_FRACTION = 0.8
MIN_OBSERVATIONS = 700
MIN_SPAN_DAYS = 1095


def load_series(path):
    frame = pd.read_csv(path)
    if not {"date", "value"} <= set(frame):
        raise ValueError("CSV requires date,value columns")
    if "ticker" in frame and frame.ticker.nunique(dropna=False) != 1:
        raise ValueError("Exactly one ticker is required; mixed series are forbidden")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    frame["value"] = pd.to_numeric(frame.value, errors="raise")
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError("Missing or duplicate dates")
    if not np.isfinite(frame.value).all() or (frame.value <= 0).any():
        raise ValueError("Values must be finite and positive")
    frame = frame.sort_values("date").reset_index(drop=True)
    if (
        len(frame) < MIN_OBSERVATIONS
        or (frame.date.iloc[-1] - frame.date.iloc[0]).days < MIN_SPAN_DAYS
    ):
        raise ValueError("At least three years and 700 daily observations required")
    if frame.date.diff().dropna().median() != pd.Timedelta(days=1):
        raise ValueError("Expected daily observations, allowing weekends and holidays")
    return frame


def prepare_series(frame, window):
    values = frame.value.to_numpy(dtype=float)
    n = len(values)
    train_end, validation_end = (
        int(n * TRAIN_END_FRACTION),
        int(n * VALIDATION_END_FRACTION),
    )
    if not 1 <= window < train_end or validation_end == train_end:
        raise ValueError("Window must fit strictly within Train")
    mean, std = float(values[:train_end].mean()), float(values[:train_end].std())
    if std == 0:
        raise ValueError("Constant Train series cannot be standardized")
    normalized = (values - mean) / std
    X = np.zeros((n, window, 1), dtype=np.float32)
    # X[t] contains t-window,...,t-1. Targets define split membership.
    for t in range(window, n):
        X[t, :, 0] = normalized[t - window : t]
    return {
        "X": X,
        "y": normalized.astype(np.float32),
        "mean": mean,
        "std": std,
        "indices": {
            "Train": np.arange(window, train_end),
            "Validation": np.arange(train_end, validation_end),
            "Test": np.arange(validation_end, n),
        },
    }
