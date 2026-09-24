"""Reproducible CPU search; select on pre-Test data and evaluate one frozen setting."""

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from diagnostics.io import sha256, write_json
from diagnostics.timeseries.data import load_series
from diagnostics.timeseries.evaluation import metrics
from diagnostics.timeseries.training import RecurrentForecaster

SPACE = {
    "window": [5, 10, 20, 30, 60],
    "hidden": [4, 8, 16, 24, 32, 64],
    "lr": [0.00003, 0.0001, 0.0003, 0.001, 0.003],
    "weight_decay": [0.0, 0.001, 0.01, 0.1],
    "batch": [32, 64, 256],
    "loss": ["mae", "mse"],
}
SEEDS = [7, 17, 42, 67, 101, 202, 340, 512, 777, 1024]
FOLDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]


def trial(job):
    config, seed, train_end, validation_end, values, epochs, test = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    window = config["window"]
    mean, std = values[:train_end].mean(), values[:train_end].std()
    # During search even the arrays stop before Test.
    limit = len(values) if test else validation_end
    z = ((values[:limit] - mean) / std).astype(np.float32)
    x = torch.from_numpy(
        np.lib.stride_tricks.sliding_window_view(z, window)[:-1].copy()
    ).unsqueeze(-1)
    y = torch.from_numpy(z[window:].copy())
    train_count, validation_count = train_end - window, validation_end - window
    vx = x[train_count:validation_count]
    actual = values[train_end:validation_end]
    naive = values[train_end - 1 : validation_end - 1]
    naive_mae = metrics(actual, naive)["MAE"]
    model = RecurrentForecaster(hidden=config["hidden"], residual=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    loss_fn = (
        nn.functional.l1_loss if config["loss"] == "mae" else nn.functional.mse_loss
    )
    # The initialized residual head is exactly Naive and is a valid fallback.
    best, best_mae, best_epoch, stale = deepcopy(model.state_dict()), naive_mae, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for start in range(0, train_count, config["batch"]):
            batch_x = x[start : min(start + config["batch"], train_count)]
            batch_y = y[start : min(start + config["batch"], train_count)]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            correction = (model(vx) - vx[:, -1, 0]).numpy().astype(float) * std
        validation_mae = metrics(actual, naive + correction)["MAE"]
        if validation_mae < best_mae - 1e-8:
            best, best_mae, best_epoch, stale = (
                deepcopy(model.state_dict()),
                validation_mae,
                epoch,
                0,
            )
        else:
            stale += 1
        if stale >= 10:
            break
    result = {
        **config,
        "seed": seed,
        "train_end": train_end,
        "validation_end": validation_end,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "naive_mae": naive_mae,
        "mae": best_mae,
        "gain_percent": 100 * (1 - best_mae / naive_mae),
    }
    if test:
        model.load_state_dict(best)
        model.eval()
        with torch.no_grad():
            tx = x[validation_count:]
            correction = (model(tx) - tx[:, -1, 0]).numpy().astype(float) * std
        prediction = values[validation_end - 1 : -1] + correction
        result.update(
            {
                "test_" + key: value
                for key, value in metrics(values[validation_end:], prediction).items()
            }
        )
        result["prediction"] = prediction.tolist()
    return result


def run_jobs(jobs, output, workers):
    rows = pd.read_csv(output).to_dict("records") if output.exists() else []
    keys = {
        (r["config_id"], r["seed"], r["train_end"], r["validation_end"]) for r in rows
    }
    total = len(jobs)
    jobs = [j for j in jobs if (j[0]["config_id"], j[1], j[2], j[3]) not in keys]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(trial, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            temporary = output.with_suffix(".tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(output)
            if len(rows) % 16 == 0 or len(rows) == total:
                print(f"{output.name}: {len(rows)}/{total} completed", flush=True)
    return pd.DataFrame(rows)


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


if __name__ == "__main__":
    main()
