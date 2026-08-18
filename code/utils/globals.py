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

# --------------------------- paths
DATASET_PATH = _resolve(config["dataset_path"], 2)
ARTIFACTS_DIR = _resolve(config.get("artifacts_dir", "artifacts"), 1)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------- data
BANDS = config.get("BANDS", [])
N_BANDS = len(BANDS)                # C, number of spectral bands

TRAIN_SOURCES = config["TRAIN_SOURCES"]
TEST_CLASSES = config["TEST_CLASSES"]

NORMAL_VAL_FRACTION = config["NORMAL_VAL_FRACTION"]
EVAL_VAL_FRACTION = config["EVAL_VAL_FRACTION"]
SPLIT_SALT = config["SPLIT_SALT"]

CLIP_REFLECTANCE = config["CLIP_REFLECTANCE"]


# --------------------------- features
PATCH_SIZE = config["PATCH_SIZE"]
N_TOKENS = PATCH_SIZE * PATCH_SIZE          # P * P = T, number of tokens in a patch
N_DCT_COEFFS = config["N_DCT_COEFFS"]

USE_RAW = config["USE_RAW"]
USE_DCT = config["USE_DCT"]

FEATURE_DIM = (N_BANDS if USE_RAW else 0) + (N_DCT_COEFFS if USE_DCT else 0)    # C + K = F, features dimension



# ======================================================= VALIDATION
if not DATASET_PATH.is_dir():
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Set dataset_path in config.yaml")

if PATCH_SIZE % 2 == 0:
    raise ValueError(f"PATCH_SIZE must be odd (patches are pixel-centered), got {PATCH_SIZE}")
if not (1 <= N_DCT_COEFFS <= N_BANDS):
    raise ValueError(f"N_DCT_COEFFS must be in [1, {N_BANDS}], got {N_DCT_COEFFS}")
if FEATURE_DIM == 0:
    raise ValueError(f"At least one of USE_RAW or USE_DCT must be True, got USE_RAW={USE_RAW}, USE_DCT={USE_DCT}")
