"""패키지로 옮긴 탐색 명령과 병렬 학습 작업의 실행 경로를 검증한다."""

import subprocess
import sys
from importlib.metadata import distribution

import numpy as np
import pandas as pd
import pytest

from diagnostics.search.execution import run_jobs


@pytest.mark.smoke
def test_search_console_entrypoint_has_no_training_side_effects(tmp_path):
    entrypoints = {
        entry.name: entry.value
        for entry in distribution("predictive-model-diagnostics-pipeline").entry_points
    }
    assert entrypoints["diagnostics-search"] == "diagnostics.search.cli:main"
    result = subprocess.run(
        [sys.executable, "-m", "diagnostics.search", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--resume" in result.stdout and "--workers" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_invalid_search_budget_preserves_workspace(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "diagnostics.search",
            "--output",
            str(tmp_path / "run"),
            "--epochs",
            "0",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert not (tmp_path / "run").exists()


def test_parallel_search_serializes_packaged_job_and_resumes_without_duplicates(
    tmp_path,
):
    config = {
        "config_id": 0,
        "window": 3,
        "hidden": 4,
        "lr": 0.001,
        "weight_decay": 0.01,
        "batch": 8,
        "loss": "mse",
    }
    values = 100 + np.arange(30, dtype=float) / 20
    job = (config, 42, 18, 24, values, 1, False)
    output = tmp_path / "search.csv"
    first = run_jobs([job], output, 1)
    assert len(first) == 1 and first.iloc[0].config_id == 0
    assert 0 <= first.iloc[0].best_epoch <= 1
    assert np.isfinite(first.iloc[0].mae)
    before = output.read_bytes()
    resumed = run_jobs([job], output, 1)
    pd.testing.assert_frame_equal(first, resumed, check_dtype=False)
    assert output.read_bytes() == before
    assert not output.with_suffix(".tmp").exists()
