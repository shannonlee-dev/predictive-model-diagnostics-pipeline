"""Values shared by command-line entry points and experiment modules."""

from pathlib import Path

DEFAULT_DATA_DIR = Path("data")
DEFAULT_REPORT_DIR = Path("reports/reference")
DEFAULT_SERIES_CSV = Path("datasets/krw_2020_2024.csv")
DEFAULT_SEED = 42
TIMESERIES_DEFAULT_EPOCHS = 50
TIMESERIES_DEFAULT_WINDOW = 30
VISION_DEFAULT_EPOCHS = 10
VISION_DEFAULT_SHOTS_PER_CLASS = 40
VISION_DEFAULT_VALIDATION_PER_CLASS = 150
VISION_DEFAULT_TEST_PER_CLASS = 250

EXCHANGE_RATE_SERIES_ID = "DEXKOUS"
EXCHANGE_RATE_SOURCE_URL = (
    "https://www.federalreserve.gov/releases/h10/hist/dat00_ko.htm"
)

VISION_MODEL_NAME = "ResNet18"
RESNET18_WEIGHTS_NAME = "IMAGENET1K_V1"
RESNET18_CHECKPOINT_PATH = "weights/checkpoints/resnet18-f37072fd.pth"

TIMESERIES_MODELS = ("RNN", "LSTM", "LSTM_residual")
VISION_STRATEGIES = ("scratch", "linear_probe", "fine_tune", "augmented")
