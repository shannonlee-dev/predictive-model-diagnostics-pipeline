"""Read and validate completed experiment artifacts without writing outputs."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import TIMESERIES_MODELS, VISION_STRATEGIES
from ..review import summarize_reviews


@dataclass
class ReportInputs:
    image_audit: dict
    series_audit: dict
    image_metrics: pd.DataFrame
    series_metrics: pd.DataFrame
    predictions: pd.DataFrame
    review_status: dict
    histories: dict
    image_predictions: dict


def read_table(path, columns, numeric=(), unique=()):
    table = pd.read_csv(path)
    if table.empty or not set(columns) <= set(table):
        raise ValueError(f"{path}: expected nonempty table with columns {columns}")
    if unique and table.duplicated(list(unique)).any():
        raise ValueError(f"{path}: duplicate keys {unique}")
    for column in numeric:
        values = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: {column} must be finite")
        table[column] = values
    return table


def load_report_inputs(root):
    root = Path(root)
    image, series = root / "vision", root / "timeseries"
    # Read both completion manifests before any output is changed.
    ia = json.loads((image / "audit.json").read_text(encoding="utf-8"))
    ta = json.loads((series / "audit.json").read_text(encoding="utf-8"))
    for audit, required in [
        (
            ia,
            (
                "seed",
                "classes",
                "shots",
                "validation_per_class",
                "test_per_class",
                "input_size",
            ),
        ),
        (ta, ("observations", "mean", "window", "selected_baseline")),
    ]:
        if not set(required) <= audit.keys():
            raise ValueError(f"Incomplete audit: required fields {required}")
    if not ia["classes"]:
        raise ValueError("Image audit must contain classes")
    im = read_table(
        image / "metrics.csv",
        ("model", "split", "loss", "accuracy"),
        ("loss", "accuracy"),
        ("model", "split"),
    )
    tm = read_table(
        series / "metrics.csv",
        ("model", "MAE", "RMSE", "MAPE"),
        ("MAE", "RMSE"),
        ("model",),
    )
    for model in VISION_STRATEGIES:
        if not {"Train", "Validation", "Test"} <= set(
            im.loc[im.model == model, "split"]
        ):
            raise ValueError(f"Missing metric splits for {model}")
    if not set(TIMESERIES_MODELS) <= set(tm.model):
        raise ValueError("Missing recurrent model metrics")
    tm["MAPE"] = pd.to_numeric(tm.MAPE, errors="raise")
    if np.isinf(tm.MAPE).any() or (tm[["MAE", "RMSE", "MAPE"]] < 0).any().any():
        raise ValueError("Metrics must be nonnegative and not infinite")
    predictions = read_table(
        series / "predictions.csv",
        ("date", "actual", "LSTM", "LSTM_residual", "Naive"),
        ("actual", "LSTM", "LSTM_residual", "Naive"),
        ("date",),
    )
    predictions["date"] = pd.to_datetime(predictions.date, errors="raise")
    if predictions.date.isna().any() or not predictions.date.is_monotonic_increasing:
        raise ValueError("Prediction dates must be present and chronological")
    histories = {}
    for track, models in [
        ("vision", im.model.unique()),
        ("timeseries", TIMESERIES_MODELS),
    ]:
        for model in models:
            table = read_table(
                root / track / f"{model}_history.csv",
                ("epoch", "Train", "Validation"),
                ("epoch", "Train", "Validation"),
                ("epoch",),
            )
            if (table[["Train", "Validation"]] < 0).any().any():
                raise ValueError("History losses must be nonnegative")
            histories[track, model] = table
    image_predictions = {}
    for model in im.model.unique():
        table = read_table(
            image / f"{model}_Test_predictions.csv",
            ("actual", "predicted"),
            ("actual", "predicted"),
        )
        valid_labels = range(len(ia["classes"]))
        if (
            not table.actual.isin(valid_labels).all()
            or not table.predicted.isin(valid_labels).all()
        ):
            raise ValueError(f"Invalid class index in {model} predictions")
        image_predictions[model] = table
    return ReportInputs(
        ia,
        ta,
        im,
        tm,
        predictions,
        summarize_reviews(image / "error_review.csv"),
        histories,
        image_predictions,
    )
