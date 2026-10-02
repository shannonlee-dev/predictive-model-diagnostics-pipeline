"""검증 구간으로만 체크포인트를 선택하는 단일 학습 작업."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from diagnostics.timeseries.evaluation import metrics
from diagnostics.timeseries.training import RecurrentForecaster


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
