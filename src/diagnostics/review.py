"""Summarize human review tags and render an error contact sheet."""

from pathlib import Path

import pandas as pd
from PIL import Image

from .io import write_json
from .plotting import plt

ALLOWED_REVIEW_TAGS = {
    "dark_lighting",
    "blur",
    "low_resolution",
    "background_clutter",
    "occlusion",
    "class_similarity",
    "suspected_label_error",
    "high_confidence_error",
    "model_limitation",
}


def summarize_reviews(path):
    frame = pd.read_csv(path).fillna("")
    required = {"sample_id", "human_tag"}
    if not required <= set(frame) or frame.sample_id.duplicated().any():
        raise ValueError("Review requires unique sample IDs and human_tag columns")
    frame["human_tag"] = frame.human_tag.str.strip()
    reviewed = frame.human_tag.isin(ALLOWED_REVIEW_TAGS)
    invalid = frame.human_tag.ne("") & ~frame.human_tag.isin(ALLOWED_REVIEW_TAGS)
    if invalid.any():
        raise ValueError("Unknown human tags")
    return {
        "total_errors": len(frame),
        "reviewed": int(reviewed.sum()),
        "human_tag_counts": frame.loc[reviewed, "human_tag"].value_counts().to_dict(),
    }


def gallery(path, model=None):
    path = Path(path)
    frame = pd.read_csv(path).fillna("")
    result = summarize_reviews(path)
    if "model" in frame and not frame.empty:
        model = frame.model.iloc[0]
    frame["human_tag"] = frame.human_tag.str.strip()
    write_json(path.parent / "review_status.json", result)
    # A static contact sheet renders directly inside the Markdown report.
    samples = frame.sort_values(["human_tag", "sample_id"]).head(30)
    fig, axes = plt.subplots(5, 6, figsize=(15, 13))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        with Image.open(path.parent / row.image) as image:
            ax.imshow(image, interpolation="nearest")
        tag = row.human_tag or "unreviewed"
        ax.set_title(
            f"{row.sample_id.rsplit('_', 1)[-1]}: {row.actual} → {row.predicted}\n{tag}",
            fontsize=9,
        )
    prefix = f"{model} — " if model else ""
    fig.suptitle(
        f"{prefix}Validation errors: {len(samples)} examples; "
        f"Tagged: {result['reviewed']}/{result['total_errors']}"
    )
    fig.tight_layout()
    fig.savefig(path.parent / "error_gallery.png", dpi=120)
    plt.close(fig)
    return result
