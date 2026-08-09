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


# GLOBALS
DATASET_PATH = Path(config["dataset_path"])
if not DATASET_PATH.is_absolute():
    DATASET_PATH = (Path(__file__).resolve().parents[2] / DATASET_PATH).resolve()

BANDS = config.get("BANDS", [])