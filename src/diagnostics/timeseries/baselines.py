"""Causal forecast baseline predictions."""

import pandas as pd

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
