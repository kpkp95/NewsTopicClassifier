from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = PROJECT_ROOT / "models" / "best_model"

DATASET_NAME = "fancyzhx/ag_news"
MODEL_NAME = "distilbert-base-uncased"

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {index: label for index, label in enumerate(LABEL_NAMES)}

RANDOM_SEED = 42
VALIDATION_SIZE = 0.10
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3
DROPOUT = 0.30
WARMUP_RATIO = 0.10
GRADIENT_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 2


def create_output_directories() -> None:
    """Create folders used for models, figures, and metrics."""
    for path in (OUTPUT_DIR, FIGURE_DIR, METRICS_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)

