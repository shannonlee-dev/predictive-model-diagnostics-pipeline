"""Single-series rolling one-step forecasts; no hidden state crosses samples."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import loss_plot, seed_everything, sha256, write_json


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
    if len(frame) < 700 or (frame.date.iloc[-1] - frame.date.iloc[0]).days < 1095:
        raise ValueError("At least three years and 700 daily observations required")
    if frame.date.diff().dropna().median() != pd.Timedelta(days=1):
        raise ValueError("Expected daily observations, allowing weekends and holidays")
    return frame


def prepare_series(frame, window):
    values = frame.value.to_numpy(dtype=float)
    n = len(values)
    train_end, validation_end = int(n * 0.7), int(n * 0.8)
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


def baseline_predictions(values):
    series = pd.Series(values)
    past = series.shift(1)  # Features constructed before slicing, always causal.
    result = {"Naive": past.to_numpy()}
    for window in (5, 10, 20):
        result[f"SMA{window}"] = past.rolling(window).mean().to_numpy()
    for alpha in (0.1, 0.3, 0.5):
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
    nonzero = np.abs(actual) > 1e-8
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "MAPE": float((np.abs(error[nonzero] / actual[nonzero])).mean() * 100)
        if nonzero.any()
        else None,
        "MAPE_excluded": int((~nonzero).sum()),
    }


class RecurrentForecaster(nn.Module):
    def __init__(self, kind="LSTM", hidden=24, residual=False):
        super().__init__()
        self.core = (nn.LSTM if kind == "LSTM" else nn.RNN)(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.residual = residual
        if residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        states, _ = self.core(x)  # Fresh zero state for each independent window.
        prediction = self.head(states[:, -1]).squeeze(-1)
        return prediction + x[:, -1, 0] if self.residual else prediction


def train_model(prepared, kind, epochs, seed, residual=False):
    seed_everything(seed)
    model = RecurrentForecaster(kind, residual=residual)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01 if residual else 0
    )
    X, y = torch.from_numpy(prepared["X"]), torch.from_numpy(prepared["y"])
    indices = prepared["indices"]
    loader = DataLoader(
        TensorDataset(X[indices["Train"]], y[indices["Train"]]),
        batch_size=64,
        shuffle=False,
    )
    best, best_loss, stale = None, float("inf"), 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, target in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(inputs), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = {
                split: float(nn.functional.mse_loss(model(X[idx]), y[idx]))
                for split, idx in indices.items()
                if split != "Test"
            }
        history.append({"epoch": epoch, **losses})
        if losses["Validation"] < best_loss:
            best_loss, best, stale = (
                losses["Validation"],
                deepcopy(model.state_dict()),
                0,
            )
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        prediction = (
            model(X[prepared["indices"]["Test"]]).numpy() * prepared["std"]
            + prepared["mean"]
        )
    return model, pd.DataFrame(history), prediction


def run(csv, output, epochs=50, seed=42, window=30):
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
    for name, kind, residual in [
        ("RNN", "RNN", False),
        ("LSTM", "LSTM", False),
        ("LSTM_residual", "LSTM", True),
    ]:
        model, history, prediction = train_model(prepared, kind, epochs, seed, residual)
        history.to_csv(output / f"{name}_history.csv", index=False)
        loss_plot(history, output / f"{name}_loss.png", name + " (standardized MSE)")
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
