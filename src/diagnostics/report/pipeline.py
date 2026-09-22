"""Coordinate analysis, rendering, and output generation."""

from pathlib import Path

from ..io import markdown_table, write_json
from ..review import gallery
from .analysis import (
    build_baseline_comparison,
    build_confusion_counts,
    build_loss_diagnosis,
    build_report_summary,
)
from .inputs import load_report_inputs
from .render import (
    build_report_context,
    render_baseline_plot,
    render_prediction_plot,
    render_template,
)

TEMPLATE_DIR = Path(__file__).with_name("templates")


def run(root):
    root = Path(root)
    image, series = root / "vision", root / "timeseries"
    data = load_report_inputs(root)
    comparison = build_baseline_comparison(data.series_metrics)
    diagnosis = build_loss_diagnosis(data.histories)
    confusion = build_confusion_counts(
        data.image_predictions, data.image_audit["classes"]
    )
    summary = build_report_summary(
        data.image_metrics,
        data.series_metrics,
        data.predictions,
        data.series_audit["mean"],
    )
    diagnosis_document = render_template(
        TEMPLATE_DIR / "diagnosis.md", build_report_context(data, diagnosis, summary)
    )
    comparison_document = render_template(
        TEMPLATE_DIR / "baseline_comparison.md",
        {
            "metrics_table": markdown_table(data.series_metrics),
            "comparison_table": markdown_table(comparison.fillna("N/A")),
        },
    )
    # All report inputs and templates have been checked before output generation.
    gallery(image / "error_review.csv")
    render_prediction_plot(data.predictions, series / "predictions.png")
    render_baseline_plot(data.series_metrics, series / "baseline_comparison.png")
    comparison.to_csv(series / "improvements_percent.csv", index=False)
    diagnosis.to_csv(root / "loss_diagnosis.csv", index=False)
    confusion.to_csv(image / "confusion_counts.csv", index=False)
    (root / "diagnosis.md").write_text(diagnosis_document, encoding="utf-8")
    (root / "baseline_comparison.md").write_text(comparison_document, encoding="utf-8")
    write_json(
        root / "status.json",
        {"experiments_complete": True, "human_review": data.review_status},
    )
    return root
