"""Values shared by command-line entry points and experiment modules."""

from pathlib import Path

DEFAULT_DATA_DIR = Path("data")
DEFAULT_REPORT_DIR = Path("reports/reference")
DEFAULT_SERIES_CSV = Path("datasets/krw_2020_2024.csv")
DEFAULT_SEED = 42
TIMESERIES_DEFAULT_EPOCHS = 50
TIMESERIES_DEFAULT_WINDOW = 30
VISION_DEFAULT_EPOCHS = 10
VISION_DEFAULT_SHOTS = 40
VISION_DEFAULT_VALIDATION = 150
VISION_DEFAULT_TEST_PER_CLASS = 250

SERIES_NAME = "DEXKOUS"
SERIES_SOURCE_URL = "https://www.federalreserve.gov/releases/h10/hist/dat00_ko.htm"

MODEL_NAME = "ResNet18"
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
PRETRAINED_CHECKPOINT = "weights/checkpoints/resnet18-f37072fd.pth"

TIMESERIES_MODELS = ("RNN", "LSTM", "LSTM_residual")
VISION_STRATEGIES = ("scratch", "linear_probe", "fine_tune", "augmented")
