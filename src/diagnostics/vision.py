"""Matched few-shot ResNet18 experiments with immutable sample memberships."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from .constants import (
    DEFAULT_SEED,
    MODEL_NAME,
    PRETRAINED_WEIGHTS,
    VISION_DEFAULT_EPOCHS,
    VISION_DEFAULT_SHOTS,
    VISION_DEFAULT_TEST_PER_CLASS,
    VISION_DEFAULT_VALIDATION,
    VISION_STRATEGIES,
)
from .io import sha256, write_json
from .plotting import loss_plot
from .reproducibility import seed_everything

CLASSES = [3, 4, 5]
NAMES = ["cat", "deer", "dog"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
INPUT_SIZE = 128
TRAIN_BATCH_SIZE = 24
EVALUATION_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 4
LINEAR_PROBE_LEARNING_RATE = 0.001
DEFAULT_LEARNING_RATE = 0.0003
AUGMENTED_WEIGHT_DECAY = 0.01
AUGMENT_BRIGHTNESS = 0.3
AUGMENT_CONTRAST = 0.2
AUGMENT_CROP_PADDING = 12
AUGMENT_FLIP_PROBABILITY = 0.5
LOW_BRIGHTNESS_THRESHOLD = 0.25
HIGH_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_EPOCHS = VISION_DEFAULT_EPOCHS
DEFAULT_SHOTS = VISION_DEFAULT_SHOTS
DEFAULT_VALIDATION = VISION_DEFAULT_VALIDATION
DEFAULT_TEST_PER_CLASS = VISION_DEFAULT_TEST_PER_CLASS
MAX_SHOTS_PER_CLASS = 49


def split_indices(labels, classes, shots, validation, seed):
    if (
        not 1 <= shots <= MAX_SHOTS_PER_CLASS
        or validation < 1
        or len(set(classes)) != len(classes)
    ):
        raise ValueError(
            f"Use 1–{MAX_SHOTS_PER_CLASS} Train images per class and nonempty Validation"
        )
    rng = np.random.default_rng(seed)
    result = {"Train": [], "Validation": []}
    for cls in classes:
        indices = np.flatnonzero(np.asarray(labels) == cls)
        if len(indices) < shots + validation:
            raise ValueError("Insufficient images for disjoint splits")
        indices = rng.permutation(indices)
        result["Train"].extend(indices[:shots].tolist())
        result["Validation"].extend(indices[shots : shots + validation].tolist())
    return result


def build_model(strategy, pretrained=True):
    model = models.resnet18(
        weights=models.ResNet18_Weights[PRETRAINED_WEIGHTS] if pretrained else None
    )
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    if strategy == "linear_probe":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    return model


def preprocessing(augment=False):
    steps = [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(p=AUGMENT_FLIP_PROBABILITY),
            transforms.ColorJitter(
                brightness=AUGMENT_BRIGHTNESS, contrast=AUGMENT_CONTRAST
            ),
            transforms.RandomCrop(
                INPUT_SIZE, padding=AUGMENT_CROP_PADDING, padding_mode="reflect"
            ),
        ]
    return transforms.Compose(
        steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


class Images(Dataset):
    def __init__(self, source, indices, augment=False):
        self.source, self.indices, self.transform = (
            source,
            indices,
            preprocessing(augment),
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = self.indices[index]
        image, label = self.source[original]
        return self.transform(image), CLASSES.index(label)


@torch.no_grad()
def evaluate(model, loader):
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
            else DEFAULT_LEARNING_RATE
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
            split: evaluate(model, loader)[0] for split, loader in evaluation.items()
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


def export_errors(source, indices, probability, labels, output):
    rows = []
    (output / "errors").mkdir()
    # Validation only: error inspection cannot leak Test into improvement decisions.
    for position in np.flatnonzero(probability.argmax(1) != labels):
        original = indices[position]
        image, _ = source[original]
        sample_id = f"cifar10_train_{original}"
        relative = f"errors/{sample_id}.png"
        image.save(output / relative)
        brightness = float(np.asarray(image).mean() / 255)
        confidence = float(probability[position].max())
        suggested = (
            "dark_lighting"
            if brightness < LOW_BRIGHTNESS_THRESHOLD
            else (
                "high_confidence_error"
                if confidence > HIGH_CONFIDENCE_THRESHOLD
                else "class_similarity"
            )
        )
        rows.append(
            {
                "sample_id": sample_id,
                "split": "Validation",
                "image": relative,
                "actual": NAMES[labels[position]],
                "predicted": NAMES[probability[position].argmax()],
                "confidence": confidence,
                "suggested_tag": suggested,
                "human_tag": "",
            }
        )
    columns = [
        "sample_id",
        "split",
        "image",
        "actual",
        "predicted",
        "confidence",
        "suggested_tag",
        "human_tag",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output / "error_review.csv", index=False)


def run(
    data,
    output,
    epochs=DEFAULT_EPOCHS,
    seed=DEFAULT_SEED,
    shots=DEFAULT_SHOTS,
    validation=DEFAULT_VALIDATION,
    test_per_class=DEFAULT_TEST_PER_CLASS,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    train_source = datasets.CIFAR10(str(data), train=True, download=False)
    test_source = datasets.CIFAR10(str(data), train=False, download=False)
    indices = split_indices(train_source.targets, CLASSES, shots, validation, seed)
    rng = np.random.default_rng(seed)
    test_indices = []
    for cls in CLASSES:
        candidates = np.flatnonzero(np.asarray(test_source.targets) == cls)
        if not 1 <= test_per_class <= len(candidates):
            raise ValueError("Invalid Test sample count")
        test_indices.extend(rng.permutation(candidates)[:test_per_class].tolist())
    membership = [
        {
            "sample_id": f"cifar10_train_{i}",
            "split": s,
            "label": NAMES[CLASSES.index(train_source.targets[i])],
        }
        for s, ids in indices.items()
        for i in ids
    ]
    membership += [
        {
            "sample_id": f"cifar10_test_{i}",
            "split": "Test",
            "label": NAMES[CLASSES.index(test_source.targets[i])],
        }
        for i in test_indices
    ]
    # Check exact image duplicates, not just numeric IDs, across selected splits.
    fingerprints = {}
    import hashlib

    for row in membership:
        is_test = row["split"] == "Test"
        original = int(row["sample_id"].rsplit("_", 1)[1])
        raw = (test_source if is_test else train_source).data[original]
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        if digest in fingerprints and fingerprints[digest] != row["split"]:
            raise ValueError("Identical image content crosses split boundaries")
        fingerprints[digest] = row["split"]
        row["sha256"] = digest
    pd.DataFrame(membership).to_csv(output / "membership.csv", index=False)
    evaluation = {
        s: DataLoader(Images(train_source, ids), batch_size=EVALUATION_BATCH_SIZE)
        for s, ids in indices.items()
    }
    test_loader = DataLoader(
        Images(test_source, test_indices), batch_size=EVALUATION_BATCH_SIZE
    )
    rows = []
    for strategy in VISION_STRATEGIES:
        seed_everything(seed)
        model = build_model(strategy, pretrained=strategy != "scratch")
        train_loader = DataLoader(
            Images(train_source, indices["Train"], strategy == "augmented"),
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
        )
        history = fit(model, strategy, train_loader, evaluation, epochs)
        history.to_csv(output / f"{strategy}_history.csv", index=False)
        loss_plot(
            history, output / f"{strategy}_loss.png", strategy + " (cross entropy)"
        )
        torch.save(model.state_dict(), output / f"{strategy}.pt")
        for split, loader in {**evaluation, "Test": test_loader}.items():
            metric, probability, labels = evaluate(model, loader)
            rows.append({"model": strategy, "split": split, **metric})
            if strategy == "fine_tune" and split == "Validation":
                export_errors(
                    train_source, indices["Validation"], probability, labels, output
                )
            prediction_frame = pd.DataFrame(
                {
                    "sample_id": [
                        f"cifar10_{'test' if split == 'Test' else 'train'}_{i}"
                        for i in (test_indices if split == "Test" else indices[split])
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
            "classes": NAMES,
            "shots": shots,
            "validation_per_class": validation,
            "test_per_class": test_per_class,
            "model": MODEL_NAME,
            "weights": PRETRAINED_WEIGHTS,
            "input_size": INPUT_SIZE,
            "mean": MEAN,
            "std": STD,
            "membership_sha256": sha256(output / "membership.csv"),
            "exact_duplicate_check": "passed",
            "linear_probe_batchnorm": "frozen",
            "improvement": "Predeclared brightness/crop/flip augmentation + weight decay; not human-reviewed causality",
        },
    )
    return output
