"""Independent-window recurrent models and validation-selected training."""

from copy import deepcopy

import pandas as pd
import torch
from torch import nn

RECURRENT_HIDDEN_SIZE = 24
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


def build_model(kind, residual=False):
    return RecurrentForecaster(kind, residual=residual)


@torch.no_grad()
def _evaluate(model, inputs, target):
    model.eval()
    return float(nn.functional.mse_loss(model(inputs), target))


@torch.no_grad()
def predict(model, inputs):
    model.eval()
    return model(inputs).numpy()


def fit(model, residual, train_loader, evaluation, epochs):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=RECURRENT_LEARNING_RATE,
        weight_decay=RESIDUAL_WEIGHT_DECAY if residual else 0,
    )
    best, best_loss, stale = None, float("inf"), 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, target in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(inputs), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
        losses = {
            split: _evaluate(model, inputs, target)
            for split, (inputs, target) in evaluation.items()
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
    return pd.DataFrame(history)


def train_model(prepared, kind, epochs, seed, residual=False):
    """Compatibility import for the original training entry point."""
    from .pipeline import train_model as pipeline_train_model

    return pipeline_train_model(prepared, kind, epochs, seed, residual)
