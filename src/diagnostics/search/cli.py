"""CPU 잔차 LSTM 탐색의 데이터 준비·선택·평가 흐름을 조립한다."""

import argparse
import json
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics.io import sha256, write_json
from diagnostics.timeseries.data import load_series
from diagnostics.timeseries.evaluation import metrics

from .config import FOLDS, SEEDS, SPACE
from .execution import run_jobs
from .reporting import append_uncertainty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("datasets/krw_2020_2024.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3600)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.trials, args.epochs, args.workers) < 1:
        parser.error("trials, epochs and workers must be positive")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and (args.output / "selected.json").exists():
        parser.error("Cannot extend search after freezing a selection or viewing Test")
    frame = load_series(args.csv)
    values = frame.value.to_numpy(dtype=float)
    n = len(values)
    train_end, validation_end = int(n * 0.7), int(n * 0.8)
    grid = [dict(zip(SPACE, combination)) for combination in product(*SPACE.values())]
    random.Random(20260924).shuffle(grid)
    baseline = dict(
        window=30, hidden=24, lr=0.001, weight_decay=0.01, batch=64, loss="mse"
    )
    configs = [baseline] + [c for c in grid if c != baseline][: args.trials - 1]
    for i, config in enumerate(configs):
        config["config_id"] = i
    protocol = {
        "dataset_sha256": sha256(args.csv),
        "search_space": SPACE,
        "trials": len(configs),
        "max_epochs": args.epochs,
        "patience": 10,
        "checkpoint_metric": "Validation MAE; initialized Naive included",
        "screen_seed": 42,
        "folds": FOLDS,
        "validation_seeds": SEEDS[:3],
        "test_seeds": SEEDS,
        "test_start": str(frame.date.iloc[validation_end].date()),
        "selection": "highest validation win fraction (>0.01% gain), then mean gain",
        "test_policy": "evaluate only the frozen selected config; do not retune after Test",
        "historical_test_note": "Test has already been inspected in the reference experiment; not a pristine holdout",
    }
    if args.resume:
        previous = json.loads((args.output / "protocol.json").read_text())
        if protocol["trials"] < previous["trials"]:
            parser.error("Cannot resume with fewer trials")
        for field in ["dataset_sha256", "max_epochs", "search_space"]:
            if previous[field] != protocol[field]:
                parser.error(f"Cannot resume with changed {field}")
    write_json(args.output / "protocol.json", protocol)
    search_values = values[:validation_end].copy()
    jobs = [
        (c, 42, train_end, validation_end, search_values, args.epochs, False)
        for c in configs
    ]
    screen = run_jobs(jobs, args.output / "search.csv", args.workers)
    shortlist = (
        screen.sort_values(["gain_percent", "config_id"], ascending=[False, True])
        .head(24)
        .config_id.tolist()
    )
    jobs = [
        (configs[i], seed, int(n * a), int(n * b), search_values, args.epochs, False)
        for i in shortlist
        for a, b in FOLDS
        for seed in SEEDS[:3]
    ]
    validation_path = args.output / "validation.csv"
    if validation_path.exists():
        previous = pd.read_csv(validation_path)
        previous[previous.config_id.isin(shortlist)].to_csv(
            validation_path, index=False
        )
    validation = run_jobs(jobs, validation_path, args.workers)
    ranking = (
        validation.assign(win=validation.gain_percent > 0.01)
        .groupby("config_id")
        .agg(
            win_fraction=("win", "mean"),
            mean_gain=("gain_percent", "mean"),
            worst_gain=("gain_percent", "min"),
        )
        .sort_values(["win_fraction", "mean_gain"], ascending=False)
    )
    chosen_id = int(ranking.index[0])
    chosen = configs[chosen_id]
    write_json(
        args.output / "selected.json",
        {"parameters": chosen, "validation": ranking.loc[chosen_id].to_dict()},
    )
    print("FROZEN SELECTION: " + json.dumps(chosen), flush=True)
    jobs = [
        (chosen, seed, train_end, validation_end, values, args.epochs, True)
        for seed in SEEDS
    ]
    results = run_jobs(jobs, args.output / "test.csv", args.workers).sort_values("seed")
    predictions = pd.DataFrame(
        {
            "date": frame.date.iloc[validation_end:].to_numpy(),
            "actual": values[validation_end:],
            "Naive": values[validation_end - 1 : -1],
        }
    )
    for _, row in results.iterrows():
        predictions[f"seed_{int(row.seed)}"] = row.prediction
    results = results.drop(columns="prediction")
    naive_metrics = metrics(predictions.actual, predictions.Naive)
    results["test_gain_percent"] = 100 * (1 - results.test_MAE / naive_metrics["MAE"])
    results.to_csv(args.output / "test.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    wins = int((results.test_gain_percent > 0.01).sum())
    selected_validation = validation[validation.config_id == chosen_id]
    # Summarize chronological quarters without selecting a new configuration.
    block_lines = []
    for i, block in enumerate(np.array_split(np.arange(len(predictions)), 4), 1):
        part = predictions.iloc[block]
        naive_mae = metrics(part.actual, part.Naive)["MAE"]
        gains = [
            100 * (1 - metrics(part.actual, part[f"seed_{seed}"])["MAE"] / naive_mae)
            for seed in SEEDS
        ]
        block_lines.append(
            f"| {i} | {part.date.iloc[0]:%Y-%m-%d}–{part.date.iloc[-1]:%Y-%m-%d} | {np.mean(gains):.4f} | {sum(g > 0.01 for g in gains)}/10 |"
        )
    document = f"""# 잔차 LSTM 하이퍼파라미터 탐색

{len(configs)}개 설정을 탐색하고 상위 {len(shortlist)}개를 3개 시간 구간 × 3개 seed로 비교했다.
Train 구간으로만 정규화하고 Validation MAE로 체크포인트를 선택했다. epoch 0의 Naive 동일 예측도 후보에 포함했다.
선택한 설정을 고정한 후 Test에서 10개 seed를 평가했다. 0.01%를 초과하는 MAE 개선을 승리로 집계한다.
기존 reference 실험에서 Test를 이미 관찰했으므로 완전히 새로운 holdout 검증은 아니다.

## 선택 설정

```json
{json.dumps(chosen, indent=2)}
```

- Validation: {int((selected_validation.gain_percent > 0.01).sum())}/9 승, 평균 개선 {selected_validation.gain_percent.mean():.4f}%
- Test: {wins}/10 승, 평균 개선 {results.test_gain_percent.mean():.4f}%, 최악 {results.test_gain_percent.min():.4f}%
- Naive Test MAE: {naive_metrics["MAE"]:.8f}

| Seed | 선택 epoch | Test MAE | Test RMSE | Test MAPE (%) | MAE 개선 (%) |
| --- | --- | --- | --- | --- | --- |
"""
    for _, row in results.iterrows():
        document += f"| {int(row.seed)} | {int(row.best_epoch)} | {row.test_MAE:.8f} | {row.test_RMSE:.8f} | {row.test_MAPE:.8f} | {row.test_gain_percent:.4f} |\n"
    document += (
        "\n## Test 시기별 일관성\n\n| 구간 | 날짜 | seed 평균 MAE 개선 (%) | 승리 seed |\n| --- | --- | --- | --- |\n"
        + "\n".join(block_lines)
    )
    document += "\n\n모든 seed의 Test 전체 성능과 시기별 성능을 함께 판단해야 한다. 미세한 개선만으로 통계적 우위를 단정하지 않는다.\n"
    (args.output / "report.md").write_text(document, encoding="utf-8")
    append_uncertainty(args.output)
    print(
        f"DONE: Test wins {wins}/10; mean gain {results.test_gain_percent.mean():.4f}%",
        flush=True,
    )
