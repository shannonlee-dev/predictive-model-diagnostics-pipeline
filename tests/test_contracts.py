import numpy as np
import pandas as pd
import pytest

from diagnostics.review import summarize_reviews
from diagnostics.timeseries import (
    baseline_predictions,
    load_series,
    metrics,
    prepare_series,
)
from diagnostics.vision import build_model, split_indices


def frame(n=100):
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "value": np.arange(n, dtype=float) + 100,
        }
    )


def test_scaler_uses_only_train_and_windows_are_strictly_past():
    data = frame()
    first = prepare_series(data, 10)
    data.loc[80:, "value"] += 10000
    changed = prepare_series(data, 10)
    assert first["mean"] == np.mean(np.arange(70) + 100)
    assert first["mean"] == changed["mean"]
    np.testing.assert_array_equal(first["X"][:71], changed["X"][:71])
    assert first["indices"]["Validation"].tolist() == list(range(70, 80))
    assert first["indices"]["Test"].tolist() == list(range(80, 100))
    np.testing.assert_allclose(
        first["X"][70, :, 0] * first["std"] + first["mean"], np.arange(60, 70) + 100
    )


def test_baselines_do_not_see_target_or_future():
    a = np.arange(100, dtype=float) + 1
    b = a.copy()
    b[80:] = 99999
    before, after = baseline_predictions(a), baseline_predictions(b)
    assert before["Naive"][80] == 80
    assert before["SMA5"][80] == 78
    for name in before:
        np.testing.assert_allclose(before[name][:81], after[name][:81], equal_nan=True)


def test_metrics_zero_target_and_negative_improvement():
    result = metrics(np.array([0.0, 2.0]), np.array([1.0, 4.0]))
    assert result["MAE"] == 1.5
    assert result["MAPE"] == 100
    assert result["MAPE_excluded"] == 1


def test_load_rejects_multiple_series_duplicates_and_short_span(tmp_path):
    for extra in ["duplicate", "ticker", "short"]:
        data = frame()
        if extra == "duplicate":
            data.loc[1, "date"] = data.loc[0, "date"]
        if extra == "ticker":
            data["ticker"] = ["A", "B"] * 50
        path = tmp_path / "series.csv"
        data.to_csv(path, index=False)
        with pytest.raises(ValueError):
            load_series(path)


def test_few_shot_reproducible_disjoint_and_balanced():
    labels = np.repeat([3, 4, 5], 300)
    a = split_indices(labels, [3, 4, 5], 40, 120, 42)
    b = split_indices(labels, [3, 4, 5], 40, 120, 42)
    assert a == b
    assert not set(a["Train"]) & set(a["Validation"])
    assert len(a["Train"]) == 120
    for cls in [3, 4, 5]:
        assert sum(labels[a["Train"]] == cls) == 40
    with pytest.raises(ValueError):
        split_indices(labels, [3, 4, 5], 50, 120, 42)


def test_linear_probe_freezes_backbone_including_batchnorm():
    model = build_model("linear_probe", pretrained=False)
    assert all(
        not p.requires_grad
        for n, p in model.named_parameters()
        if not n.startswith("fc.")
    )
    assert all(p.requires_grad for p in model.fc.parameters())


def test_human_review_not_inferred_from_suggested_tags(tmp_path):
    path = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "1",
                "suggested_tag": "low_resolution",
                "human_tag": "",
            }
        ]
    ).to_csv(path, index=False)
    result = summarize_reviews(path)
    assert result["reviewed"] == 0
    assert result["human_tag_counts"] == {}
    assert "complete" not in result


def test_recurrent_predictions_do_not_share_sample_context():
    import torch

    from diagnostics.timeseries import RecurrentForecaster

    torch.manual_seed(42)
    model = RecurrentForecaster().eval()
    x = torch.randn(2, 30, 1)
    with torch.no_grad():
        together = model(x)
        alone = model(x[1:])
        altered = x.clone()
        altered[0] += 1000
        changed = model(altered)
    torch.testing.assert_close(together[1:], alone)
    torch.testing.assert_close(together[1:], changed[1:])


def test_residual_model_starts_at_naive():
    import torch

    from diagnostics.timeseries import RecurrentForecaster

    model = RecurrentForecaster(residual=True)
    x = torch.randn(5, 30, 1)
    torch.testing.assert_close(model(x), x[:, -1, 0])


@pytest.mark.parametrize("count", [1, 5, 30, 31])
def test_review_counts_are_informational(tmp_path, count):
    path = tmp_path / "review.csv"
    rows = [
        {
            "sample_id": str(i),
            "human_tag": "occlusion",
        }
        for i in range(count)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    assert summarize_reviews(path) == {
        "total_errors": count,
        "reviewed": count,
        "human_tag_counts": {"occlusion": count},
    }
    rows[-1]["human_tag"] = ""
    pd.DataFrame(rows).to_csv(path, index=False)
    assert summarize_reviews(path)["reviewed"] == count - 1
    rows.append(rows[0].copy())
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError):
        summarize_reviews(path)


def test_invalid_series_and_window_fail_instead_of_silent_repair(tmp_path):
    data = frame(1200)
    data.loc[5, "value"] = np.inf
    path = tmp_path / "series.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_series(path)
    with pytest.raises(ValueError):
        prepare_series(frame(), 70)


def test_short_training_produces_finite_test_predictions():
    from diagnostics.timeseries import train_model

    _model, history, prediction = train_model(
        prepare_series(frame(), 10), "LSTM", 2, 42
    )
    assert len(prediction) == 20
    assert np.isfinite(prediction).all()
    assert history.Train.notna().all()


def test_linear_probe_training_preserves_batchnorm_statistics():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from diagnostics.vision import fit

    torch.set_num_threads(2)
    model = build_model("linear_probe", pretrained=False)
    before = model.bn1.running_mean.clone()
    weights = model.conv1.weight.detach().clone()
    data = DataLoader(
        TensorDataset(torch.randn(3, 3, 64, 64), torch.tensor([0, 1, 2])), batch_size=3
    )
    fit(model, "linear_probe", data, {"Train": data, "Validation": data}, 1)
    torch.testing.assert_close(model.bn1.running_mean, before)
    torch.testing.assert_close(model.conv1.weight, weights)


def test_monthly_observations_cannot_pass_as_daily(tmp_path):
    data = frame(750)
    data["date"] = pd.date_range("1960-01-01", periods=750, freq="MS")
    path = tmp_path / "monthly.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="daily"):
        load_series(path)
