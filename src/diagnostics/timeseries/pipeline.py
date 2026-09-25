"""Coordinate rolling forecast experiments and persist their artifacts."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..constants import (
    DEFAULT_SEED,
    TIMESERIES_DEFAULT_EPOCHS,
    TIMESERIES_DEFAULT_WINDOW,
)
from ..io import sha256, write_json
from ..plotting import loss_plot
from ..reproducibility import seed_everything
from .baselines import baseline_predictions
from .data import load_series, prepare_series
from .evaluation import metrics
from .training import build_model, fit, predict

DEFAULT_EPOCHS = TIMESERIES_DEFAULT_EPOCHS
DEFAULT_WINDOW = TIMESERIES_DEFAULT_WINDOW
TRAIN_BATCH_SIZE = 64
RECURRENT_BATCH_SIZE = TRAIN_BATCH_SIZE  # Preserve the public constant.


def _training_data(prepared):
    X = torch.from_numpy(prepared["X"])
    y = torch.from_numpy(prepared["y"])
    indices = prepared["indices"]
    train_loader = DataLoader(
        TensorDataset(X[indices["Train"]], y[indices["Train"]]),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
    )
    evaluation = {
        split: (X[idx], y[idx]) for split, idx in indices.items() if split != "Test"
    }
    return X, indices, train_loader, evaluation


def train_model(prepared, kind, epochs, seed, residual=False):
    """Preserve the original training API for existing callers."""
    X, indices, train_loader, evaluation = _training_data(prepared)
    seed_everything(seed)
    model = build_model(kind, residual=residual)
    history = fit(model, train_loader, evaluation, epochs)
    prediction = predict(model, X[indices["Test"]]) * prepared["std"] + prepared["mean"]
    return model, history, prediction


def run(
    csv,
    output,
    epochs=DEFAULT_EPOCHS,
    seed=DEFAULT_SEED,
    window=DEFAULT_WINDOW,
    save_checkpoints=False,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frame = load_series(csv)
    prepared = prepare_series(frame, window)
    idx = prepared["indices"]["Test"]
    actual = frame.value.to_numpy()[idx]
    predictions = pd.DataFrame(
        {"date": frame.date.iloc[idx].to_numpy(), "actual": actual}
    )
    rows = []
    baselines = baseline_predictions(frame.value.to_numpy())
    for name, values in baselines.items():
        predictions[name] = values[idx]
        rows.append({"model": name, **metrics(actual, values[idx])})
    # Persist evaluated baselines before any neural-network training.
    pd.DataFrame(rows).to_csv(output / "baselines_before_training.csv", index=False)
    validation = prepared["indices"]["Validation"]
    baseline_validation = {
        n: metrics(frame.value.to_numpy()[validation], p[validation])["MAE"]
        for n, p in baselines.items()
    }
    selected_baseline = min(baseline_validation, key=baseline_validation.get)
    X, indices, train_loader, evaluation = _training_data(prepared)
    for name, kind, residual in [
        ("RNN", "RNN", False),
        ("LSTM", "LSTM", False),
        ("LSTM_residual", "LSTM", True),
    ]:
        seed_everything(seed)
        model = build_model(kind, residual=residual)
        history = fit(model, train_loader, evaluation, epochs)
        prediction = (
            predict(model, X[indices["Test"]]) * prepared["std"] + prepared["mean"]
        )
        history.to_csv(output / f"{name}_history.csv", index=False)
        loss_plot(history, output / f"{name}_loss.png", name + " (standardized MSE)")
        if save_checkpoints:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "kind": kind,
                    "residual": residual,
                    "mean": prepared["mean"],
                    "std": prepared["std"],
                    "window": window,
                },
                output / f"{name}.pt",
            )
        predictions[name] = prediction
        rows.append({"model": name, **metrics(actual, prediction)})
        print(f"{name}: {rows[-1]}", flush=True)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    audit = {
        split: {
            "first_target": str(frame.date.iloc[ids[0]].date()),
            "last_target": str(frame.date.iloc[ids[-1]].date()),
            "count": len(ids),
        }
        for split, ids in prepared["indices"].items()
    }
    write_json(
        output / "audit.json",
        {
            "seed": seed,
            "epochs": epochs,
            "window": window,
            "sha256": sha256(csv),
            "observations": len(frame),
            "mean": prepared["mean"],
            "std": prepared["std"],
            "splits": audit,
            "selected_baseline": selected_baseline,
            "baseline_validation_mae": baseline_validation,
            "features": "shift(1) before rolling; input ends at target-1",
            "protocol": "rolling one-step; observed past Test values allowed; no refit; independent hidden states",
        },
    )
    return output
