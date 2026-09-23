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
    EXCHANGE_RATE_SERIES_ID,
    EXCHANGE_RATE_SOURCE_URL,
    RESNET18_WEIGHTS_NAME,
    VISION_MODEL_NAME,
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


def _retry(operation):
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
    _retry(lambda: datasets.CIFAR10(str(root), train=True, download=True))
    _retry(lambda: datasets.CIFAR10(str(root), train=False, download=True))
    _retry(
        lambda: models.resnet18(weights=models.ResNet18_Weights[RESNET18_WEIGHTS_NAME])
    )
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
            "source": EXCHANGE_RATE_SOURCE_URL,
            "series": EXCHANGE_RATE_SERIES_ID,
            **metadata,
            "sha256": sha256(root / "exchange.csv"),
            "image_source": "torchvision.datasets.CIFAR10; official archive checksum verified by torchvision",
            "model": VISION_MODEL_NAME,
            "weights": RESNET18_WEIGHTS_NAME,
        },
    )
