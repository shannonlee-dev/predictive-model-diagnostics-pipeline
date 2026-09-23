"""Calculations on measured experiment tables; no file I/O or rendering."""

import math

import pandas as pd

from ..constants import TIMESERIES_MODELS

PERCENT_SCALE = 100
HIGH_VARIANCE_RATIO = 1.5
ERROR_METRIC_COLUMNS = ("MAE", "RMSE", "MAPE")


def _improvement_percent(baseline, measured):
    """Undefined improvements (zero or missing baseline) remain missing."""
    if pd.isna(baseline) or pd.isna(measured) or baseline == 0:
        return float("nan")
    return PERCENT_SCALE * (baseline - measured) / baseline


def build_baseline_comparison(metrics, neural_models=TIMESERIES_MODELS):
    rows = metrics.set_index("model")
    baselines = [model for model in metrics.model if model not in neural_models]
    return pd.DataFrame(
        [
            {
                "model": model,
                "baseline": baseline,
                **{
                    metric: _improvement_percent(
                        rows.loc[baseline, metric], rows.loc[model, metric]
                    )
                    for metric in ERROR_METRIC_COLUMNS
                },
            }
            for model in neural_models
            for baseline in baselines
        ],
        columns=["model", "baseline", *ERROR_METRIC_COLUMNS],
    )


def build_loss_diagnosis(histories, ratio_threshold=HIGH_VARIANCE_RATIO):
    diagnoses = []
    for (_, model), history in histories.items():
        row = history.loc[history.Validation.idxmin()]
        ratio = (
            row.Validation / row.Train
            if row.Train != 0
            else math.inf
            if row.Validation > 0
            else 1.0
        )
        diagnoses.append(
            {
                "model": model,
                "best_epoch": int(row.epoch),
                "Train_loss": row.Train,
                "Validation_loss": row.Validation,
                "gap": row.Validation - row.Train,
                "interpretation": (
                    "High Variance 또는 분포 이동 의심"
                    if ratio > ratio_threshold
                    else "큰 일반화 격차 없음; 절대 오차·베이스라인과 함께 편향 판단"
                ),
            }
        )
    return pd.DataFrame(diagnoses)


def build_confusion_counts(predictions, classes):
    rows = []
    for strategy, table in predictions.items():
        for actual, actual_name in enumerate(classes):
            for predicted, predicted_name in enumerate(classes):
                rows.append(
                    {
                        "model": strategy,
                        "actual": actual_name,
                        "predicted": predicted_name,
                        "count": int(
                            (
                                (table.actual == actual)
                                & (table.predicted == predicted)
                            ).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_report_summary(image_metrics, series_metrics, predictions, train_mean):
    """Select by Validation loss and calculate unformatted measured comparisons."""
    image_test = image_metrics[image_metrics.split == "Test"].set_index("model")
    series_test = series_metrics.set_index("model")
    selected = (
        image_metrics[image_metrics.split == "Validation"]
        .sort_values("loss")
        .iloc[0]
        .model
    )
    lstm = series_test.loc["LSTM", "MAE"]
    residual = series_test.loc["LSTM_residual", "MAE"]
    return {
        "selected_strategy": selected,
        "transfer_gain": (
            image_test.loc["fine_tune", "accuracy"]
            - image_test.loc["scratch", "accuracy"]
        )
        * PERCENT_SCALE,
        "augmentation_gain": (
            image_test.loc["augmented", "accuracy"]
            - image_test.loc["fine_tune", "accuracy"]
        )
        * PERCENT_SCALE,
        "residual_gain": _improvement_percent(lstm, residual),
        "rnn_mae": series_test.loc["RNN", "MAE"],
        "lstm_mae": lstm,
        "test_train_drift": predictions.actual.mean() - train_mean,
    }
