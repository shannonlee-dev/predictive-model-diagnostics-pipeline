"""Few-shot membership, duplicate checks, and image transforms."""

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms

SELECTED_CIFAR10_CLASS_IDS = [3, 4, 5]
SELECTED_CIFAR10_CLASS_NAMES = ["cat", "deer", "dog"]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMAGE_INPUT_SIZE = 128
AUGMENT_BRIGHTNESS = 0.3
AUGMENT_CONTRAST = 0.2
AUGMENT_CROP_PADDING = 12
AUGMENT_FLIP_PROBABILITY = 0.5
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


def _preprocessing(augment=False):
    steps = [transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(p=AUGMENT_FLIP_PROBABILITY),
            transforms.ColorJitter(
                brightness=AUGMENT_BRIGHTNESS, contrast=AUGMENT_CONTRAST
            ),
            transforms.RandomCrop(
                IMAGE_INPUT_SIZE, padding=AUGMENT_CROP_PADDING, padding_mode="reflect"
            ),
        ]
    return transforms.Compose(
        steps
        + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


class Images(Dataset):
    def __init__(self, source, indices, augment=False):
        self.source, self.indices, self.transform = (
            source,
            indices,
            _preprocessing(augment),
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = self.indices[index]
        image, label = self.source[original]
        return self.transform(image), SELECTED_CIFAR10_CLASS_IDS.index(label)


@dataclass
class PreparedVisionData:
    train_source: Dataset
    test_source: Dataset
    indices: dict[str, list[int]]
    test_indices: list[int]
    membership: pd.DataFrame


def prepare_data(data, shots, validation, test_per_class, seed):
    """Select disjoint samples and audit exact duplicates before training."""
    train_source = datasets.CIFAR10(str(data), train=True, download=False)
    test_source = datasets.CIFAR10(str(data), train=False, download=False)
    indices = split_indices(
        train_source.targets, SELECTED_CIFAR10_CLASS_IDS, shots, validation, seed
    )
    rng = np.random.default_rng(seed)
    test_indices = []
    for cls in SELECTED_CIFAR10_CLASS_IDS:
        candidates = np.flatnonzero(np.asarray(test_source.targets) == cls)
        if not 1 <= test_per_class <= len(candidates):
            raise ValueError("Invalid Test sample count")
        test_indices.extend(rng.permutation(candidates)[:test_per_class].tolist())
    membership = [
        {
            "sample_id": f"cifar10_train_{i}",
            "split": s,
            "label": SELECTED_CIFAR10_CLASS_NAMES[
                SELECTED_CIFAR10_CLASS_IDS.index(train_source.targets[i])
            ],
        }
        for s, ids in indices.items()
        for i in ids
    ]
    membership += [
        {
            "sample_id": f"cifar10_test_{i}",
            "split": "Test",
            "label": SELECTED_CIFAR10_CLASS_NAMES[
                SELECTED_CIFAR10_CLASS_IDS.index(test_source.targets[i])
            ],
        }
        for i in test_indices
    ]
    # Check exact image duplicates, not just numeric IDs, across selected splits.
    fingerprints = {}
    for row in membership:
        is_test = row["split"] == "Test"
        original = int(row["sample_id"].rsplit("_", 1)[1])
        raw = (test_source if is_test else train_source).data[original]
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        if digest in fingerprints and fingerprints[digest] != row["split"]:
            raise ValueError("Identical image content crosses split boundaries")
        fingerprints[digest] = row["split"]
        row["sha256"] = digest
    return PreparedVisionData(
        train_source=train_source,
        test_source=test_source,
        indices=indices,
        test_indices=test_indices,
        membership=pd.DataFrame(membership),
    )
