"""Exercise experiment wiring without downloads or expensive vision backbones."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torch import nn

from diagnostics import timeseries, vision
from diagnostics.io import sha256


@pytest.fixture
def offline_vision(monkeypatch):
    class Samples:
        def __init__(self, root, train, download):
            assert download is False
            self.targets = np.repeat([3, 4, 5], 4).tolist()
            self.data = np.random.default_rng(int(train)).integers(
                0, 256, (12, 32, 32, 3), dtype=np.uint8
            )

        def __getitem__(self, index):
            return Image.fromarray(self.data[index]), self.targets[index]

    class SmallClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)

        def forward(self, images):
            return self.fc(images.mean(dim=(2, 3)))

    monkeypatch.setattr("torchvision.datasets.CIFAR10", Samples)
    monkeypatch.setattr(
        "torchvision.models.resnet18", lambda **kwargs: SmallClassifier()
    )
    return Samples


@pytest.mark.parametrize("save_checkpoints", [False, True])
def test_vision_pipeline_exports_matched_splits_and_review(
    tmp_path, offline_vision, save_checkpoints
):
    output = tmp_path / "vision"
    assert (
        vision.run(
            tmp_path,
            output,
            epochs=1,
            shots=1,
            validation=1,
            test_per_class=1,
            save_checkpoints=save_checkpoints,
        )
        == output
    )
    membership = pd.read_csv(output / "membership.csv")
    assert membership.groupby("split").size().to_dict() == {
        "Train": 3,
        "Validation": 3,
        "Test": 3,
    }
    assert membership.sample_id.is_unique
    assert membership.sha256.is_unique
    metrics = pd.read_csv(output / "metrics.csv")
    assert len(metrics) == 12
    assert np.isfinite(metrics[["accuracy", "loss"]]).all().all()
    for strategy in metrics.model.unique():
        assert (output / f"{strategy}.pt").is_file() == save_checkpoints
        history = pd.read_csv(output / f"{strategy}_history.csv")
        assert list(history) == ["epoch", "Train", "Validation"]
        for split in ["Train", "Validation"]:
            assert not (output / f"{strategy}_{split}_predictions.csv").exists()
        for split in ["Test"]:
            predictions = pd.read_csv(output / f"{strategy}_{split}_predictions.csv")
            assert (
                predictions.sample_id.tolist()
                == membership.loc[membership.split == split, "sample_id"].tolist()
            )
    review = pd.read_csv(output / "error_review.csv")
    assert "suggested_tag" not in review.columns
    assert set(review.sample_id) <= set(
        membership.loc[membership.split == "Validation", "sample_id"]
    )
    assert review.human_tag.isna().all()
    assert all((output / image).is_file() for image in review.image)
    assert not (output / "errors.zip").exists()
    audit = json.loads((output / "audit.json").read_text())
    assert audit["membership_sha256"] == sha256(output / "membership.csv")
    with pytest.raises(FileExistsError):
        vision.run(tmp_path, output, epochs=1)


def test_vision_rejects_duplicate_content_before_training(
    tmp_path, offline_vision, monkeypatch
):
    original = offline_vision.__init__

    def identical_samples(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.data[:] = 0

    monkeypatch.setattr(offline_vision, "__init__", identical_samples)
    output = tmp_path / "vision"
    with pytest.raises(ValueError, match="Identical image content"):
        vision.run(
            tmp_path,
            output,
            epochs=1,
            shots=1,
            validation=1,
            test_per_class=1,
        )
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("save_checkpoints", [False, True])
def test_timeseries_pipeline_exports_predictions_and_audit(tmp_path, save_checkpoints):
    csv = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=1200),
            "value": 100 + np.arange(1200) / 20 + np.sin(np.arange(1200)),
        }
    ).to_csv(csv, index=False)
    output = tmp_path / "timeseries"
    assert (
        timeseries.run(csv, output, epochs=1, save_checkpoints=save_checkpoints)
        == output
    )
    predictions = pd.read_csv(output / "predictions.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    baselines = pd.read_csv(output / "baselines_before_training.csv")
    assert len(predictions) == 240
    assert set(predictions.columns) == {"date", "actual", *metrics.model}
    pd.testing.assert_frame_equal(metrics.iloc[: len(baselines)], baselines)
    for name in ["RNN", "LSTM", "LSTM_residual"]:
        assert (output / f"{name}.pt").is_file() == save_checkpoints
        actual = predictions.actual.to_numpy()
        expected = timeseries.metrics(actual, predictions[name].to_numpy())
        assert metrics.set_index("model").loc[name, "MAE"] == pytest.approx(
            expected["MAE"]
        )
    audit = json.loads((output / "audit.json").read_text())
    assert audit["sha256"] == sha256(csv)
    assert audit["splits"]["Test"]["count"] == len(predictions)
    assert audit["selected_baseline"] == min(
        audit["baseline_validation_mae"], key=audit["baseline_validation_mae"].get
    )
    with pytest.raises(FileExistsError):
        timeseries.run(csv, output, epochs=1)
