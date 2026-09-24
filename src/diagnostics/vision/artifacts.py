"""Export validation errors for optional human review."""

import numpy as np
import pandas as pd

from .data import SELECTED_CIFAR10_CLASS_NAMES

LOW_BRIGHTNESS_THRESHOLD = 0.25
HIGH_CONFIDENCE_THRESHOLD = 0.8


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
                "actual": SELECTED_CIFAR10_CLASS_NAMES[labels[position]],
                "predicted": SELECTED_CIFAR10_CLASS_NAMES[
                    probability[position].argmax()
                ],
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
