import argparse
import logging
import os
from pathlib import Path


def positive(value):
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
    prepare.add_argument("--data", type=Path, default=Path("data"))
    for name in ["timeseries", "vision"]:
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--epochs", type=positive, default=50 if name == "timeseries" else 10
        )
        command.add_argument("--seed", type=int, default=42)
        if name == "timeseries":
            command.add_argument(
                "--csv", type=Path, default=Path("datasets/krw_2020_2024.csv")
            )
            command.add_argument("--window", type=positive, default=30)
        else:
            command.add_argument("--data", type=Path, default=Path("data"))
            command.add_argument("--shots", type=positive, default=40)
            command.add_argument("--validation", type=positive, default=150)
            command.add_argument("--test-per-class", type=positive, default=250)
    report = sub.add_parser("report")
    report.add_argument("--run", type=Path, default=Path("reports/reference"))
    review = sub.add_parser("review")
    review.add_argument("--csv", type=Path, required=True)
    review.add_argument("--require-complete", action="store_true")
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
            if not (
                args["data"] / "weights/checkpoints/resnet18-f37072fd.pth"
            ).is_file():
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
            if args["require_complete"] and not result["complete"]:
                raise ValueError(
                    "At least 30 unique, explicitly human-reviewed errors required"
                )
    except (ValueError, OSError, RuntimeError) as error:
        logging.getLogger(__name__).error(
            "%s: %s; incomplete output directories are not completed experiments",
            name,
            error,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
