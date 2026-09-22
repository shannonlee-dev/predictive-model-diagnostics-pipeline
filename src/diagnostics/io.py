"""Small file and tabular serialization helpers."""

import hashlib
import json
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_table(frame):
    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row)
            + " |"
        )
    return "\n".join(rows)
