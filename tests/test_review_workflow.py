import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from diagnostics.cli import main
from diagnostics.review import summarize_reviews


@pytest.fixture
def review_csv(tmp_path):
    path = tmp_path / "error_review.csv"
    Image.new("RGB", (8, 8)).save(tmp_path / "error.png")
    pd.DataFrame(
        [
            dict(
                sample_id=f"sample_{i}",
                image="error.png",
                actual="cat",
                predicted="dog",
                confidence=0.9,
                suggested_tag="occlusion",
                human_tag="",
            )
            for i in range(31)
        ]
    ).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("count", [0, 1, 5, 30, 31])
def test_review_cli_succeeds_without_threshold(review_csv, count, monkeypatch, capsys):
    frame = pd.read_csv(review_csv).fillna("")
    frame.loc[frame.index < count, "human_tag"] = "occlusion"
    frame.to_csv(review_csv, index=False)
    monkeypatch.setattr("sys.argv", ["diagnostics", "review", "--csv", str(review_csv)])
    main()
    assert f"'reviewed': {count}" in capsys.readouterr().out
    status = json.loads((review_csv.parent / "review_status.json").read_text())
    assert status == dict(
        total_errors=31,
        reviewed=count,
        human_tag_counts={"occlusion": count} if count else {},
    )
    html = (review_csv.parent / "error_gallery.html").read_text()
    assert f"Human reviewed: {count}/31" in html
    assert "required" not in html
    assert (review_csv.parent / "error_gallery.png").is_file()


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_human_tag_is_not_counted(review_csv, blank):
    frame = pd.read_csv(review_csv).fillna("")
    frame.loc[0, "human_tag"] = blank
    frame.to_csv(review_csv, index=False)
    assert summarize_reviews(review_csv)["reviewed"] == 0
    assert summarize_reviews(review_csv)["human_tag_counts"] == {}


@pytest.mark.parametrize("invalid", ["duplicate", "tag"])
def test_review_cli_rejects_invalid_data(review_csv, invalid, monkeypatch):
    frame = pd.read_csv(review_csv).fillna("")
    if invalid == "duplicate":
        frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    else:
        frame.loc[0, "human_tag"] = "unknown"
    frame.to_csv(review_csv, index=False)
    monkeypatch.setattr("sys.argv", ["diagnostics", "review", "--csv", str(review_csv)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1


@pytest.mark.parametrize("count", [0, 5])
def test_report_completion_depends_on_tracks(tmp_path, monkeypatch, count):
    reference = Path(__file__).resolve().parents[1] / "reports/reference"
    root = tmp_path / "run"
    shutil.copytree(reference, root, ignore=shutil.ignore_patterns("*.pt"))
    path = root / "vision/error_review.csv"
    frame = pd.read_csv(path).fillna("")
    frame["human_tag"] = ""
    frame.loc[frame.index < count, "human_tag"] = "occlusion"
    frame.to_csv(path, index=False)
    monkeypatch.setattr("sys.argv", ["diagnostics", "report", "--run", str(root)])
    main()
    status = json.loads((root / "status.json").read_text())
    assert status["experiments_complete"] is True
    assert status["human_review"]["reviewed"] == count
    assert "complete" not in status["human_review"]
    assert "submission_ready" not in status
    assert "선택적인 오류 분석" in (root / "diagnosis.md").read_text()
    (root / "timeseries/audit.json").unlink()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
