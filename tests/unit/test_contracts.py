import numpy as np
import pandas as pd
import pytest

from diagnostics.constants import DEFAULT_SEED
from diagnostics.prepare import _series_metadata
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


def test_human_review_only_counts_explicit_tags(tmp_path):
    path = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "1",
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


def test_timeseries_build_model_selects_recurrent_kind():
    import torch

    from diagnostics.timeseries import build_model as build_timeseries_model

    rnn = build_timeseries_model("RNN")
    lstm = build_timeseries_model("LSTM", residual=True)
    assert isinstance(rnn.core, torch.nn.RNN)
    assert isinstance(lstm.core, torch.nn.LSTM)
    assert lstm.residual is True


def test_timeseries_fit_restores_best_validation_state():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from diagnostics.timeseries import build_model as build_timeseries_model
    from diagnostics.timeseries import fit as fit_timeseries
    from diagnostics.timeseries.training import _evaluate, predict

    torch.manual_seed(42)
    inputs = torch.randn(8, 10, 1)
    target = torch.randn(8)
    loader = DataLoader(TensorDataset(inputs, target), batch_size=4, shuffle=False)
    model = build_timeseries_model("LSTM")
    history = fit_timeseries(
        model,
        loader,
        {"Train": (inputs, target), "Validation": (inputs, target)},
        2,
    )
    assert list(history) == ["epoch", "Train", "Validation"]
    assert len(history) == 2
    assert _evaluate(model, inputs, target) == pytest.approx(history.Validation.min())
    assert np.isfinite(predict(model, inputs)).all()


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


def test_series_metadata_is_derived_from_observed_dates():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-06", "2020-01-08"]
            ),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert _series_metadata(data) == {
        "start": "2020-01-01",
        "end": "2020-01-08",
        "observations": 4,
        "dropped_missing": 2,
    }


def test_default_seed_is_shared_by_cli_and_experiments():
    from diagnostics.cli import DEFAULT_SEED as cli_seed
    from diagnostics.timeseries.pipeline import DEFAULT_SEED as timeseries_seed

    assert cli_seed == timeseries_seed == DEFAULT_SEED == 42


@pytest.mark.parametrize("kind", ["invalid", "LTSM", "lstm", ""])
def test_recurrent_model_rejects_unsupported_kind(kind):
    from diagnostics.timeseries import RecurrentForecaster
    from diagnostics.timeseries import build_model as build_timeseries_model

    for constructor in (RecurrentForecaster, build_timeseries_model):
        with pytest.raises(ValueError, match="kind"):
            constructor(kind)


def test_vision_build_model_rejects_unsupported_strategy():
    with pytest.raises(ValueError, match="strategy"):
        build_model("invalid", pretrained=False)


def test_vision_fit_rejects_unsupported_strategy():
    from diagnostics.vision import fit

    model = build_model("scratch", pretrained=False)
    with pytest.raises(ValueError, match="strategy"):
        fit(model, "invalid", [], {}, 1)


@pytest.mark.parametrize("residual, decay", [(False, 0.0), (True, 0.01)])
def test_timeseries_fit_regularization_follows_model(residual, decay):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from diagnostics.timeseries import RecurrentForecaster, fit

    # Zero loss leaves only AdamW decay: one step must scale nonzero weights.
    model = RecurrentForecaster("RNN", residual=residual)
    inputs = torch.zeros(2, 3, 1)
    target = model(inputs).detach()
    before = model.core.weight_ih_l0.detach().clone()
    loader = DataLoader(TensorDataset(inputs, target), batch_size=2)
    history = fit(model, loader, {"Validation": (inputs, target)}, 1)
    torch.testing.assert_close(
        model.core.weight_ih_l0, before * (1 - 0.001 * decay), rtol=0, atol=1e-8
    )
    assert len(history) == 1


def test_timeseries_training_does_not_import_pipeline():
    import ast
    import inspect

    from diagnostics.timeseries import training

    tree = ast.parse(inspect.getsource(training))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "pipeline" not in (node.module or "").split(".")
            assert all(alias.name != "pipeline" for alias in node.names)
        elif isinstance(node, ast.Import):
            assert all("pipeline" not in alias.name.split(".") for alias in node.names)
