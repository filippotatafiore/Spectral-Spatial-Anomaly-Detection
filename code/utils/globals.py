from pathlib import Path
import yaml

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


# --------------------------- data
BANDS = config.get("BANDS", [])
N_BANDS = len(BANDS)

TRAIN_SOURCES = config["TRAIN_SOURCES"]
TEST_CLASSES = config["TEST_CLASSES"]

NORMAL_VAL_FRACTION = config["NORMAL_VAL_FRACTION"]
EVAL_VAL_FRACTION = config["EVAL_VAL_FRACTION"]
SPLIT_SALT = config["SPLIT_SALT"]

CLIP_REFLECTANCE = config["CLIP_REFLECTANCE"]


# --------------------------- features
N_DCT_COEFFS = config["N_DCT_COEFFS"]