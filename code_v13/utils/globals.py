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
ANOMAL_VAL_FRACTION = config["ANOMAL_VAL_FRACTION"]
ANOMAL_TEST_FRACTION = config.get("ANOMAL_TEST_FRACTION", 0.20)

SPLIT_SALT = config["SPLIT_SALT"]
GROUPED_SPLIT = config.get("GROUPED_SPLIT", False)

CLIP_REFLECTANCE = config["CLIP_REFLECTANCE"]

MAX_SHIFT = int(config.get("MAX_SHIFT", 0))


# --------------------------- model
BACKBONE = config.get("BACKBONE", "cnn")
INPUT_UPSCALE = int(config.get("INPUT_UPSCALE", 1))

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

ENCODER_LR_MULT = float(config.get("ENCODER_LR_MULT", 1.0))
WARMUP_EPOCHS = int(config.get("WARMUP_EPOCHS", 0))
TYPE_LOSS_WEIGHT = float(config.get("TYPE_LOSS_WEIGHT", 0.0))

NUM_WORKERS = config["NUM_WORKERS"]
SOURCE_BALANCE_POWER = float(config.get("SOURCE_BALANCE_POWER", 0.0))
SEG_POS_WEIGHT = float(config.get("SEG_POS_WEIGHT", 8.0))

SOURCE_WEIGHT_MULT = config.get("SOURCE_WEIGHT_MULT") or {}
EARLY_STOP_PATIENCE = int(config.get("EARLY_STOP_PATIENCE", 0))

SEED = config["SEED"]

MODEL_VERSION = config.get("MODEL_VERSION", "v6")
MODEL_RUN = config.get("MODEL_RUN", "0")


# --------------------------- normality prototypes
NUM_PROTOTYPES = int(config.get("NUM_PROTOTYPES", 64))
PROTO_DIM = int(config.get("PROTO_DIM", 128))
PROTO_MOMENTUM = float(config.get("PROTO_MOMENTUM", 0.99))
DEV_MARGIN = float(config.get("DEV_MARGIN", 0.5))
DEV_TOPK = int(config.get("DEV_TOPK", 8))
DEV_LOSS_WEIGHT = float(config.get("DEV_LOSS_WEIGHT", 0.0))
PROTO_RESEED_MIN_RATE = float(config.get("PROTO_RESEED_MIN_RATE", 0.0))


# --------------------------- cross-region anomaly transplantation / coarse labels / open-set
CRAT_PROB = float(config.get("CRAT_PROB", 0.0))
CRAT_MIN_COVER = float(config.get("CRAT_MIN_COVER", 0.02))
CRAT_MAX_COVER = float(config.get("CRAT_MAX_COVER", 0.7))
CRAT_GAIN_CLAMP = float(config.get("CRAT_GAIN_CLAMP", 1.5))
CRAT_MIN_DONORS = int(config.get("CRAT_MIN_DONORS", 1))
CRAT_MIN_NDVI_DROP = float(config.get("CRAT_MIN_NDVI_DROP", 0.0))

SEG_IGNORE_BORDER = int(config.get("SEG_IGNORE_BORDER", 0))

HOLDOUT_CLASS = config.get("HOLDOUT_CLASS")


# --------------------------- test
TARGET_RECALL = float(config.get("TARGET_RECALL", 0.95))
CLS_THRESHOLD_MODE = config.get("CLS_THRESHOLD_MODE", "recall")
TARGET_FPR = float(config.get("TARGET_FPR", 0.15))



# ======================================================= VALIDATION
if not DATASET_PATH.is_dir():
    raise FileNotFoundError(
        f"Dataset not found at {DATASET_PATH}, "
        "if using kaggle, add:  os.environ[\"SSAD_DATA_ROOT\"] = \"/kaggle/input/sentinel2-anomaly-detection\""
    )