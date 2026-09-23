import argparse
import logging
import os
from pathlib import Path

from .constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_SEED,
    DEFAULT_SERIES_CSV,
    RESNET18_CHECKPOINT_PATH,
    TIMESERIES_DEFAULT_EPOCHS,
    TIMESERIES_DEFAULT_WINDOW,
    VISION_DEFAULT_EPOCHS,
    VISION_DEFAULT_SHOTS_PER_CLASS,
    VISION_DEFAULT_TEST_PER_CLASS,
    VISION_DEFAULT_VALIDATION_PER_CLASS,
)


def _positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Predictive-model diagnostics; outputs must be new directories"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    for name in ["timeseries", "vision"]:
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--epochs",
            type=_positive,
            default=(
                TIMESERIES_DEFAULT_EPOCHS
                if name == "timeseries"
                else VISION_DEFAULT_EPOCHS
            ),
        )
        command.add_argument("--seed", type=int, default=DEFAULT_SEED)
        if name == "timeseries":
            command.add_argument("--csv", type=Path, default=DEFAULT_SERIES_CSV)
            command.add_argument(
                "--window", type=_positive, default=TIMESERIES_DEFAULT_WINDOW
            )
        else:
            command.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
            command.add_argument(
                "--shots", type=_positive, default=VISION_DEFAULT_SHOTS_PER_CLASS
            )
            command.add_argument(
                "--validation",
                type=_positive,
                default=VISION_DEFAULT_VALIDATION_PER_CLASS,
            )
            command.add_argument(
                "--test-per-class",
                type=_positive,
                default=VISION_DEFAULT_TEST_PER_CLASS,
            )
    report = sub.add_parser("report")
    report.add_argument("--run", type=Path, default=DEFAULT_REPORT_DIR)
    review = sub.add_parser("review")
    review.add_argument("--csv", type=Path, required=True)
    args = vars(parser.parse_args())
    name = args.pop("command")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    logging.basicConfig(level=logging.INFO)
    try:
        if name == "prepare":
            from .prepare import run

            run(args["data"])
        elif name == "timeseries":
            from .timeseries import run

            run(**args)
        elif name == "vision":
            import torch

            torch.hub.set_dir(str(args["data"] / "weights"))
            # Prevent implicit internet access in offline experiments.
            if not (args["data"] / RESNET18_CHECKPOINT_PATH).is_file():
                raise FileNotFoundError(
                    "Run prepare first: pretrained ResNet18 weights missing"
                )
            from .vision import run

            run(**args)
        elif name == "report":
            from .report import run

            run(args["run"])
        else:
            from .review import gallery

            result = gallery(args["csv"])
            print(result)
    except (ValueError, OSError, RuntimeError) as error:
        logging.getLogger(__name__).error(
            "%s: %s; incomplete output directories are not completed experiments",
            name,
            error,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
