"""Markdown and chart presentation for report analysis results."""

import re
from pathlib import Path

from ..constants import TIMESERIES_MODELS
from ..io import markdown_table
from .analysis import HIGH_VARIANCE_RATIO

TOKEN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z_0-9]*)\s*\}\}")


def render_template(path, context):
    """Substitute each token once; context values are never interpreted as templates."""

    def replace(match):
        name = match.group(1)
        if name not in context:
            raise ValueError(f"Missing report template value: {name}")
        return str(context[name])

    return TOKEN.sub(replace, Path(path).read_text(encoding="utf-8"))


def format_percent(value):
    return "N/A" if value != value else f"{value:.2f}"


def build_report_context(data, diagnosis, summary):
    image_audit = data.image_audit
    series_audit = data.series_audit
    image_metrics = data.image_metrics
    series_metrics = data.series_metrics
    return {
        "seed": image_audit["seed"],
        "shots": image_audit["shots"],
        "validation_per_class": image_audit["validation_per_class"],
        "test_per_class": image_audit["test_per_class"],
        "observations": series_audit["observations"],
        "reviewed": data.review_status["reviewed"],
        "total_errors": data.review_status["total_errors"],
        "classes": ", ".join(image_audit["classes"]),
        "input_size": image_audit["input_size"],
        "window": series_audit["window"],
        "ratio_threshold": HIGH_VARIANCE_RATIO,
        "image_metrics_table": markdown_table(image_metrics),
        "selected_strategy": summary["selected_strategy"],
        "transfer_gain": f"{summary['transfer_gain']:.2f}",
        "augmentation_gain": f"{summary['augmentation_gain']:.2f}",
        "loss_diagnosis_table": markdown_table(diagnosis),
        "loss_images": "\n".join(
            [
                *[
                    f"![{model} Train/Validation loss](vision/{model}_loss.png)"
                    for model in image_metrics.model.unique()
                ],
                *[
                    f"![{model} Train/Validation loss](timeseries/{model}_loss.png)"
                    for model in TIMESERIES_MODELS
                ],
            ]
        ),
        "series_metrics_table": markdown_table(series_metrics),
        "residual_gain": format_percent(summary["residual_gain"]),
        "selected_baseline": series_audit["selected_baseline"],
        "rnn_mae": f"{summary['rnn_mae']:.4f}",
        "lstm_mae": f"{summary['lstm_mae']:.4f}",
        "test_train_drift": f"{summary['test_train_drift']:.2f}",
    }


PREDICTION_COLORS = {
    "LSTM": "tab:red",
    "LSTM_residual": "tab:green",
    "Naive": "gray",
}


def render_prediction_plot(prediction, output):
    from ..plotting import plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prediction.date, prediction.actual, label="Actual", color="tab:blue")
    for model, color in PREDICTION_COLORS.items():
        ax.plot(prediction.date, prediction[model], label=model, alpha=0.8, color=color)
    ax.set(xlabel="Date", ylabel="KRW per USD", title="Rolling one-step Test forecasts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_baseline_plot(metrics, output):
    from ..plotting import plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        ax.bar(metrics.model, metrics[metric])
        ax.set(ylabel=metric, title=f"Test {metric} (KRW per USD)")
        ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
