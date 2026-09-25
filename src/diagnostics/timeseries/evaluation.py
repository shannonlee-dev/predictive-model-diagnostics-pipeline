"""Regression metrics for time-series forecasts."""

import numpy as np

MAPE_ZERO_THRESHOLD = 1e-8


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
