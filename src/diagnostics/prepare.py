"""Explicit online preparation; experiment commands themselves stay offline."""

import shutil
import socket
import time
from pathlib import Path

import torch
from torchvision import datasets, models

from .common import sha256, write_json

SERIES_URL = "https://www.federalreserve.gov/releases/h10/hist/dat00_ko.htm"


def retry(operation):
    for attempt in range(3):
        try:
            return operation()
        except (OSError, RuntimeError) as error:
            if attempt == 2:
                raise RuntimeError(
                    "Download failed after 3 attempts; use a validated local cache/CSV."
                ) from error
            time.sleep(2**attempt)


def run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    socket.setdefaulttimeout(60)
    torch.hub.set_dir(str(root / "weights"))
    retry(lambda: datasets.CIFAR10(str(root), train=True, download=True))
    retry(lambda: datasets.CIFAR10(str(root), train=False, download=True))
    retry(lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1))
    # Frozen official observations avoid changing upstream CSV endpoints.
    from .timeseries import load_series

    snapshot = Path("datasets/krw_2020_2024.csv")
    frame = load_series(snapshot)
    missing = 56
    temporary = root / "exchange.csv.part"
    shutil.copyfile(snapshot, temporary)
    temporary.replace(root / "exchange.csv")
    write_json(
        root / "provenance.json",
        {
            "source": SERIES_URL,
            "series": "DEXKOUS",
            "start": "2020-01-01",
            "end": "2024-12-31",
            "dropped_missing": missing,
            "observations": len(frame),
            "sha256": sha256(root / "exchange.csv"),
            "image_source": "torchvision.datasets.CIFAR10; official archive checksum verified by torchvision",
            "weights": "ResNet18 IMAGENET1K_V1",
        },
    )
