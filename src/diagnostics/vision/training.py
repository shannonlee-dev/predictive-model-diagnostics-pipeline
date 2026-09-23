"""ResNet construction, fitting, and classification evaluation."""

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models

from ..constants import RESNET18_WEIGHTS_NAME
from .data import SELECTED_CIFAR10_CLASS_IDS

EARLY_STOPPING_PATIENCE = 4
LINEAR_PROBE_LEARNING_RATE = 0.001
FULL_MODEL_LEARNING_RATE = 0.0003
AUGMENTED_WEIGHT_DECAY = 0.01


def build_model(strategy, pretrained=True):
    model = models.resnet18(
        weights=models.ResNet18_Weights[RESNET18_WEIGHTS_NAME] if pretrained else None
    )
    model.fc = nn.Linear(model.fc.in_features, len(SELECTED_CIFAR10_CLASS_IDS))
    if strategy == "linear_probe":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    return model


@torch.no_grad()
def _evaluate(model, loader):
    model.eval()
    probabilities, labels, loss = [], [], 0.0
    for x, y in loader:
        logits = model(x)
        loss += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
        probabilities.append(logits.softmax(-1).numpy())
        labels.append(y.numpy())
    probabilities, labels = np.concatenate(probabilities), np.concatenate(labels)
    return (
        {
            "loss": loss / len(labels),
            "accuracy": float((probabilities.argmax(1) == labels).mean()),
        },
        probabilities,
        labels,
    )


def fit(model, strategy, train_loader, evaluation, epochs):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=(
            LINEAR_PROBE_LEARNING_RATE
            if strategy == "linear_probe"
            else FULL_MODEL_LEARNING_RATE
        ),
        weight_decay=AUGMENTED_WEIGHT_DECAY if strategy == "augmented" else 0.0,
    )
    history, best, best_loss, stale = [], None, float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        if strategy == "linear_probe":
            model.eval()  # Freeze BatchNorm running statistics as well as weights.
            model.fc.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        measurements = {
            split: _evaluate(model, loader)[0] for split, loader in evaluation.items()
        }
        history.append(
            {"epoch": epoch, **{k: v["loss"] for k, v in measurements.items()}}
        )
        print(f"{strategy} epoch {epoch}: {measurements}", flush=True)
        if measurements["Validation"]["loss"] < best_loss:
            best_loss, best, stale = (
                measurements["Validation"]["loss"],
                deepcopy(model.state_dict()),
                0,
            )
        else:
            stale += 1
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    model.load_state_dict(best)
    return pd.DataFrame(history)
