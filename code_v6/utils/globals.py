import torch
from pathlib import Path
import yaml
import os

def _load_config():
    """
        Load the global variables from the config.yaml file.
    """
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = _load_config()

def _resolve(path_str, parents_up=2):
    path = Path(path_str)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[parents_up] / path).resolve()
    return path



# ======================================================= GLOBALS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- paths
_env_root = os.environ.get("SSAD_DATA_ROOT")
DATASET_PATH = Path(_env_root) if _env_root else _resolve(config["dataset_path"], 2)
ARTIFACTS_DIR = _resolve(config.get("artifacts_dir", "artifacts"), 1)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------- data
TRAIN_SOURCES = config["TRAIN_SOURCES"]
TEST_CLASSES = config["TEST_CLASSES"]

NORMAL_VAL_FRACTION = config["NORMAL_VAL_FRACTION"]
NORMAL_TEST_FRACTION = config.get("NORMAL_TEST_FRACTION", 0.10)
EVAL_VAL_FRACTION = config["EVAL_VAL_FRACTION"]

SPLIT_SALT = config["SPLIT_SALT"]
GROUPED_SPLIT = config.get("GROUPED_SPLIT", False)

CLIP_REFLECTANCE = config["CLIP_REFLECTANCE"]


# --------------------------- model
IN_CHANNELS = config["IN_CHANNELS"]
EMBED_DIM = config["EMBED_DIM"]
NUM_ATT_HEADS = config["NUM_ATT_HEADS"]
NUM_TRANS_LAYERS = config["NUM_TRANS_LAYERS"]
DROPOUT = config["DROPOUT"]


# --------------------------- train
EPOCHS = config["EPOCHS"]
BATCH_IMAGES = config["BATCH_IMAGES"]

LR = float(config["LR"])
WEIGHT_DECAY = float(config["WEIGHT_DECAY"])
AMP = config["AMP"]

NUM_WORKERS = config["NUM_WORKERS"]
SOURCE_BALANCE_POWER = float(config.get("SOURCE_BALANCE_POWER", 0.0))

SEED = config["SEED"]



# ======================================================= VALIDATION
if not DATASET_PATH.is_dir():
    raise FileNotFoundError(
        f"Dataset not found at {DATASET_PATH}, "
        "if using kaggle, add:  os.environ[\"SSAD_DATA_ROOT\"] = \"/kaggle/input/sentinel2-anomaly-detection\""
    )