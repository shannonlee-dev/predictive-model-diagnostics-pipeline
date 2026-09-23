"""Causal forecast baselines and regression metrics."""

import numpy as np
import pandas as pd

MAPE_ZERO_THRESHOLD = 1e-8
BASELINE_SMA_WINDOWS = (5, 10, 20)
BASELINE_EMA_ALPHAS = (0.1, 0.3, 0.5)


def baseline_predictions(values):
    series = pd.Series(values)
    past = series.shift(1)  # Features constructed before slicing, always causal.
    result = {"Naive": past.to_numpy()}
    for window in BASELINE_SMA_WINDOWS:
        result[f"SMA{window}"] = past.rolling(window).mean().to_numpy()
    for alpha in BASELINE_EMA_ALPHAS:
        result[f"EMA{alpha}"] = past.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    return result


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if (
        actual.shape != predicted.shape
        or not np.isfinite(predicted).all()
        or not np.isfinite(actual).all()
        or not actual.size
    ):
        raise ValueError("Metrics require aligned nonempty finite arrays")
    error = actual - predicted
    nonzero = np.abs(actual) > MAPE_ZERO_THRESHOLD
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "MAPE": float((np.abs(error[nonzero] / actual[nonzero])).mean() * 100)
        if nonzero.any()
        else None,
        "MAPE_excluded": int((~nonzero).sum()),
    }
