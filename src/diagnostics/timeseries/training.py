"""Independent-window recurrent models and validation-selected training."""

from copy import deepcopy

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..reproducibility import seed_everything

RECURRENT_HIDDEN_SIZE = 24
RECURRENT_BATCH_SIZE = 64
RECURRENT_LEARNING_RATE = 0.001
RESIDUAL_WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 8


class RecurrentForecaster(nn.Module):
    def __init__(self, kind="LSTM", hidden=RECURRENT_HIDDEN_SIZE, residual=False):
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
        model.parameters(),
        lr=RECURRENT_LEARNING_RATE,
        weight_decay=RESIDUAL_WEIGHT_DECAY if residual else 0,
    )
    X, y = torch.from_numpy(prepared["X"]), torch.from_numpy(prepared["y"])
    indices = prepared["indices"]
    loader = DataLoader(
        TensorDataset(X[indices["Train"]], y[indices["Train"]]),
        batch_size=RECURRENT_BATCH_SIZE,
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
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
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
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        prediction = (
            model(X[prepared["indices"]["Test"]]).numpy() * prepared["std"]
            + prepared["mean"]
        )
    return model, pd.DataFrame(history), prediction
