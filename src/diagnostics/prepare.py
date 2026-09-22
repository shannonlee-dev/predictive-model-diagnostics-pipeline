"""Explicit online preparation; experiment commands themselves stay offline."""

import shutil
import socket
import time
from pathlib import Path

import pandas as pd
import torch
from torchvision import datasets, models

from .constants import (
    DEFAULT_SERIES_CSV,
    MODEL_NAME,
    PRETRAINED_WEIGHTS,
    SERIES_NAME,
    SERIES_SOURCE_URL,
)
from .io import sha256, write_json

RETRY_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 60
RETRY_BACKOFF_BASE = 2


def series_metadata(frame):
    """Return provenance facts calculated from the validated observations."""
    expected = pd.bdate_range(frame.date.iloc[0], frame.date.iloc[-1])
    return {
        "start": frame.date.iloc[0].date().isoformat(),
        "end": frame.date.iloc[-1].date().isoformat(),
        "observations": len(frame),
        "dropped_missing": len(expected) - len(frame),
    }


def retry(operation):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return operation()
        except (OSError, RuntimeError) as error:
            if attempt == RETRY_ATTEMPTS - 1:
                raise RuntimeError(
                    f"Download failed after {RETRY_ATTEMPTS} attempts; "
                    "use a validated local cache/CSV."
                ) from error
            time.sleep(RETRY_BACKOFF_BASE**attempt)


def run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    socket.setdefaulttimeout(REQUEST_TIMEOUT_SECONDS)
    torch.hub.set_dir(str(root / "weights"))
    retry(lambda: datasets.CIFAR10(str(root), train=True, download=True))
    retry(lambda: datasets.CIFAR10(str(root), train=False, download=True))
    retry(lambda: models.resnet18(weights=models.ResNet18_Weights[PRETRAINED_WEIGHTS]))
    # Frozen official observations avoid changing upstream CSV endpoints.
    from .timeseries import load_series

    snapshot = DEFAULT_SERIES_CSV
    frame = load_series(snapshot)
    metadata = series_metadata(frame)
    temporary = root / "exchange.csv.part"
    shutil.copyfile(snapshot, temporary)
    temporary.replace(root / "exchange.csv")
    write_json(
        root / "provenance.json",
        {
            "source": SERIES_SOURCE_URL,
            "series": SERIES_NAME,
            **metadata,
            "sha256": sha256(root / "exchange.csv"),
            "image_source": "torchvision.datasets.CIFAR10; official archive checksum verified by torchvision",
            "model": MODEL_NAME,
            "weights": PRETRAINED_WEIGHTS,
        },
    )
