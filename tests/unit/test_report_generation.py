import pandas as pd
import pytest

from diagnostics.report.analysis import build_baseline_comparison, build_report_summary
from diagnostics.report.render import render_template


@pytest.mark.parametrize("lstm_mae,expected_gain", [(10.0, 20.0), (0.0, None)])
def test_summary_selects_by_validation_and_calculates_measured_differences(
    lstm_mae, expected_gain
):
    image_metrics = pd.DataFrame(
        [
            {"model": "scratch", "split": "Validation", "loss": 0.1, "accuracy": 0.5},
            {"model": "fine_tune", "split": "Validation", "loss": 0.2, "accuracy": 0.7},
            {"model": "augmented", "split": "Validation", "loss": 0.3, "accuracy": 0.6},
            {"model": "scratch", "split": "Test", "loss": 0.9, "accuracy": 0.5},
            {"model": "fine_tune", "split": "Test", "loss": 0.01, "accuracy": 0.7},
            {"model": "augmented", "split": "Test", "loss": 0.4, "accuracy": 0.6},
        ]
    )
    series_metrics = pd.DataFrame(
        [
            {"model": "RNN", "MAE": 12.0},
            {"model": "LSTM", "MAE": lstm_mae},
            {"model": "LSTM_residual", "MAE": 8.0},
        ]
    )
    summary = build_report_summary(
        image_metrics, series_metrics, pd.DataFrame({"actual": [90.0, 100.0]}), 100.0
    )
    assert summary["selected_strategy"] == "scratch"
    assert summary["transfer_gain"] == pytest.approx(20.0)
    assert summary["augmentation_gain"] == pytest.approx(-10.0)
    assert summary["test_train_drift"] == -5.0
    assert summary["rnn_mae"] == 12.0
    assert summary["lstm_mae"] == lstm_mae
    if expected_gain is None:
        assert pd.isna(summary["residual_gain"])
    else:
        assert summary["residual_gain"] == expected_gain


def test_undefined_improvements_and_regression_remain_visible():
    metrics = pd.DataFrame(
        [
            {"model": "Naive", "MAE": 0.0, "RMSE": 10.0, "MAPE": None},
            {"model": "RNN", "MAE": 1.0, "RMSE": 15.0, "MAPE": 2.0},
        ]
    )
    row = build_baseline_comparison(metrics, ("RNN",)).iloc[0]
    assert pd.isna(row.MAE)
    assert row.RMSE == -50
    assert pd.isna(row.MAPE)


def test_template_values_are_not_recursively_substituted(tmp_path):
    path = tmp_path / "template.md"
    path.write_text("{{title}} {{ body }}", encoding="utf-8")
    assert (
        render_template(path, {"title": "{{ body }}", "body": "text"})
        == "{{ body }} text"
    )
    with pytest.raises(ValueError, match="body"):
        render_template(path, {"title": "title"})


def test_baseline_comparison_is_calculated_from_measured_metrics():
    metrics = pd.DataFrame(
        [
            {"model": "Naive", "MAE": 10.0, "RMSE": 20.0, "MAPE": 30.0},
            {"model": "RNN", "MAE": 8.0, "RMSE": 18.0, "MAPE": 27.0},
        ]
    )

    result = build_baseline_comparison(metrics, neural_models=("RNN",))

    assert result.to_dict("records") == [
        {
            "model": "RNN",
            "baseline": "Naive",
            "MAE": 20.0,
            "RMSE": 10.0,
            "MAPE": 10.0,
        }
    ]


def test_template_renderer_replaces_context_without_leaving_tokens(tmp_path):
    template = tmp_path / "report.md"
    template.write_text("# {{ title }}\n\n{{ body }}\n")

    result = render_template(template, {"title": "Report", "body": "Measured"})

    assert result == "# Report\n\nMeasured\n"
