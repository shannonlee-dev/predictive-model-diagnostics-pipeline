"""동일 Test 날짜의 불확실성과 seed별 결과를 기록한다."""

import numpy as np
import pandas as pd

from diagnostics.timeseries.evaluation import metrics

from .config import SEEDS


def append_uncertainty(output):
    """Paired moving-block bootstrap over dates, averaging per-seed absolute errors."""
    frame = pd.read_csv(output / "predictions.csv")
    columns = [f"seed_{seed}" for seed in SEEDS]
    actual = frame.actual.to_numpy()
    naive_error = np.abs(actual - frame.Naive.to_numpy())
    model_error = np.abs(actual[:, None] - frame[columns].to_numpy()).mean(axis=1)
    rng = np.random.default_rng(20260924)
    n, block_size = len(frame), 10
    starts = rng.integers(0, n, size=(5000, (n + block_size - 1) // block_size))
    indices = ((starts[:, :, None] + np.arange(block_size)) % n).reshape(5000, -1)[
        :, :n
    ]
    gains = 100 * (
        1 - model_error[indices].mean(axis=1) / naive_error[indices].mean(axis=1)
    )
    lower, upper = np.quantile(gains, [0.025, 0.975])
    paragraph = (
        "\n## 개선의 불확실성\n\n"
        f"날짜 순서의 상관을 고려한 10일 블록 bootstrap 5,000회에서, "
        f"seed 평균 MAE 개선률의 95% 구간은 **{lower:.4f}% ~ {upper:.4f}%**다. "
        "각 seed의 절대오차를 평균한 값이며 앙상블 예측 성능은 아니다. "
        "고정된 모델의 평가 날짜 불확실성만 추정하며, 탐색에 따른 선택 편향이나 미래 분포 변화는 포함하지 않는다.\n"
    )
    paragraph += (
        "0을 포함하므로 Naive보다 통계적으로 우수하다고 판단할 근거가 부족하다.\n"
        if lower <= 0 <= upper
        else "이 구간과 별도로 시기별 결과 및 새로운 기간의 검증이 필요하다.\n"
    )
    scores = pd.read_csv(output / "test.csv")
    baseline = metrics(actual, frame.Naive.to_numpy())
    paragraph += "\n| 지표 | Naive | 잔차 LSTM seed 평균 | 0.01% 초과 개선 seed |\n| --- | --- | --- | --- |\n"
    for metric in ["MAE", "RMSE", "MAPE"]:
        measured = scores[f"test_{metric}"]
        wins = int((measured < baseline[metric] * 0.9999).sum())
        paragraph += f"| {metric} | {baseline[metric]:.8f} | {measured.mean():.8f} | {wins}/10 |\n"
    paragraph += "\n여러 seed는 동일 Test 날짜를 공유하므로 독립된 10개 시장 표본을 뜻하지 않는다.\n"
    with (output / "report.md").open("a", encoding="utf-8") as report:
        report.write(paragraph)
