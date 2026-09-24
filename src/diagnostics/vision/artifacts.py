"""Export validation errors for optional human review."""

import numpy as np
import pandas as pd

from .data import SELECTED_CIFAR10_CLASS_NAMES


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
        confidence = float(probability[position].max())
        rows.append(
            {
                "sample_id": sample_id,
                "split": "Validation",
                "image": relative,
                "actual": SELECTED_CIFAR10_CLASS_NAMES[labels[position]],
                "predicted": SELECTED_CIFAR10_CLASS_NAMES[
                    probability[position].argmax()
                ],
                "confidence": confidence,
                "human_tag": "",
                "review_note": "",
            }
        )
    columns = [
        "sample_id",
        "split",
        "image",
        "actual",
        "predicted",
        "confidence",
        "human_tag",
        "review_note",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output / "error_review.csv", index=False)
