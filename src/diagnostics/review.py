"""Human review is explicit; suggested machine tags never count as review."""

import html
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


def gallery(path):
    path = Path(path)
    frame = pd.read_csv(path).fillna("")
    result = summarize_reviews(path)
    frame["human_tag"] = frame.human_tag.str.strip()
    write_json(path.parent / "review_status.json", result)
    cards = []
    for _, r in frame.sort_values(
        ["human_tag", "suggested_tag", "sample_id"]
    ).iterrows():
        tag = r.human_tag or f"UNREVIEWED · suggestion: {r.suggested_tag}"
        cards.append(
            f'<figure><img src="{html.escape(r.image, quote=True)}" width="160" height="160"><figcaption>{html.escape(r.sample_id)}<br>{html.escape(r.actual)} → {html.escape(r.predicted)} ({r.confidence:.3f})<br>{html.escape(tag)}</figcaption></figure>'
        )
    document = '<!doctype html><meta charset="utf-8"><title>Validation error review</title><style>body{font:16px sans-serif;max-width:1200px;margin:40px auto}main{display:flex;flex-wrap:wrap}figure{width:240px;margin:10px}img{image-rendering:pixelated}figcaption{overflow-wrap:anywhere}</style>'
    document += (
        f"<h1>Validation error review</h1><p>Human reviewed: {result['reviewed']}/{result['total_errors']}. Machine suggestions are hypotheses, not verified causes.</p><main>"
        + "".join(cards)
        + "</main>"
    )
    (path.parent / "error_gallery.html").write_text(document)
    # A static contact sheet also renders directly inside the Markdown report.
    samples = frame.sort_values(["human_tag", "suggested_tag", "sample_id"]).head(30)
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
    fig.suptitle(
        f"Validation errors: {len(samples)} examples; "
        f"Human reviewed: {result['reviewed']}/{result['total_errors']}"
    )
    fig.tight_layout()
    fig.savefig(path.parent / "error_gallery.png", dpi=120)
    plt.close(fig)
    frame.suggested_tag.value_counts().rename_axis("suggested_tag").to_csv(
        path.parent / "suggested_tag_counts.csv", header=["count"]
    )
    return result
