"""Coordinate matched vision experiments and persist their artifacts."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..constants import (
    DEFAULT_SEED,
    RESNET18_WEIGHTS_NAME,
    VISION_DEFAULT_EPOCHS,
    VISION_DEFAULT_SHOTS_PER_CLASS,
    VISION_DEFAULT_TEST_PER_CLASS,
    VISION_DEFAULT_VALIDATION_PER_CLASS,
    VISION_MODEL_NAME,
    VISION_STRATEGIES,
)
from ..io import sha256, write_json
from ..plotting import loss_plot
from ..reproducibility import seed_everything
from .artifacts import export_errors
from .data import (
    IMAGE_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SELECTED_CIFAR10_CLASS_NAMES,
    Images,
    prepare_data,
)
from .training import build_model, evaluate, fit

TRAIN_BATCH_SIZE = 24
EVALUATION_BATCH_SIZE = 32
DEFAULT_EPOCHS = VISION_DEFAULT_EPOCHS
DEFAULT_SHOTS_PER_CLASS = VISION_DEFAULT_SHOTS_PER_CLASS
DEFAULT_VALIDATION_PER_CLASS = VISION_DEFAULT_VALIDATION_PER_CLASS
DEFAULT_TEST_PER_CLASS = VISION_DEFAULT_TEST_PER_CLASS


def run(
    data,
    output,
    epochs=DEFAULT_EPOCHS,
    seed=DEFAULT_SEED,
    shots=DEFAULT_SHOTS_PER_CLASS,
    validation=DEFAULT_VALIDATION_PER_CLASS,
    test_per_class=DEFAULT_TEST_PER_CLASS,
    save_checkpoints=False,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    prepared = prepare_data(data, shots, validation, test_per_class, seed)
    prepared.membership.to_csv(output / "membership.csv", index=False)
    evaluation = {
        s: DataLoader(
            Images(prepared.train_source, ids), batch_size=EVALUATION_BATCH_SIZE
        )
        for s, ids in prepared.indices.items()
    }
    test_loader = DataLoader(
        Images(prepared.test_source, prepared.test_indices),
        batch_size=EVALUATION_BATCH_SIZE,
    )
    rows = []
    for strategy in VISION_STRATEGIES:
        seed_everything(seed)
        model = build_model(strategy, pretrained=strategy != "scratch")
        train_loader = DataLoader(
            Images(
                prepared.train_source,
                prepared.indices["Train"],
                strategy == "augmented",
            ),
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
        )
        history = fit(model, strategy, train_loader, evaluation, epochs)
        history.to_csv(output / f"{strategy}_history.csv", index=False)
        loss_plot(
            history, output / f"{strategy}_loss.png", strategy + " (cross entropy)"
        )
        if save_checkpoints:
            torch.save(model.state_dict(), output / f"{strategy}.pt")
        for split, loader in {**evaluation, "Test": test_loader}.items():
            metric, probability, labels = evaluate(model, loader)
            rows.append({"model": strategy, "split": split, **metric})
            if strategy == "fine_tune" and split == "Validation":
                export_errors(
                    prepared.train_source,
                    prepared.indices["Validation"],
                    probability,
                    labels,
                    output,
                )
            if split != "Test":
                continue
            prediction_frame = pd.DataFrame(
                {
                    "sample_id": [
                        f"cifar10_{'test' if split == 'Test' else 'train'}_{i}"
                        for i in (
                            prepared.test_indices
                            if split == "Test"
                            else prepared.indices[split]
                        )
                    ],
                    "actual": labels,
                    "predicted": probability.argmax(1),
                    "confidence": probability.max(1),
                }
            )
            prediction_frame.to_csv(
                output / f"{strategy}_{split}_predictions.csv", index=False
            )
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    write_json(
        output / "audit.json",
        {
            "seed": seed,
            "epochs": epochs,
            "classes": SELECTED_CIFAR10_CLASS_NAMES,
            "shots": shots,
            "validation_per_class": validation,
            "test_per_class": test_per_class,
            "model": VISION_MODEL_NAME,
            "weights": RESNET18_WEIGHTS_NAME,
            "input_size": IMAGE_INPUT_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "membership_sha256": sha256(output / "membership.csv"),
            "exact_duplicate_check": "passed",
            "linear_probe_batchnorm": "frozen",
            "improvement": "Predeclared brightness/crop/flip augmentation + weight decay; not human-reviewed causality",
        },
    )
    return output
