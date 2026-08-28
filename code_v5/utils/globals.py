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
_env_root = os.environ.get("SSAD_DATA_ROOT")
DATASET_PATH = Path(_env_root) if _env_root else _resolve(config["dataset_path"], 2)
ARTIFACTS_DIR = _resolve(config.get("artifacts_dir", "artifacts"), 1)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------- data
BANDS = config.get("BANDS", [])
N_BANDS = len(BANDS)                # C, number of spectral bands

TRAIN_SOURCES = config["TRAIN_SOURCES"]
TEST_CLASSES = config["TEST_CLASSES"]
LOCAL_CLASSES = config.get("LOCAL_CLASSES", [])
SCENE_CLASSES = config.get("SCENE_CLASSES", [])

NORMAL_VAL_FRACTION = config["NORMAL_VAL_FRACTION"]
NORMAL_TEST_FRACTION = config.get("NORMAL_TEST_FRACTION", 0.10)
EVAL_VAL_FRACTION = config["EVAL_VAL_FRACTION"]
SPLIT_SALT = config["SPLIT_SALT"]
GROUPED_SPLIT = config.get("GROUPED_SPLIT", False)

CLIP_REFLECTANCE = config["CLIP_REFLECTANCE"]


# --------------------------- features
PATCH_SIZE = config["PATCH_SIZE"]
MULTISCALE_OUTER = int(config.get("MULTISCALE_OUTER", 0))
MULTISCALE_STRIDE = int(config.get("MULTISCALE_STRIDE", 4))

def _token_offsets():
    """
        (row, col) offsets of every token, relative to the top-left of the padded window.

        Dense core: all P*P positions, as before.
        Sparse ring: positions of an OUTER x OUTER neighbourhood on a stride grid, excluding anything in the dense core.
    """
    r = PATCH_SIZE // 2
    pts = [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1)]
    if MULTISCALE_OUTER > PATCH_SIZE and MULTISCALE_STRIDE > 0:
        R = MULTISCALE_OUTER // 2
        for dr in range(-R, R + 1, MULTISCALE_STRIDE):
            for dc in range(-R, R + 1, MULTISCALE_STRIDE):
                if max(abs(dr), abs(dc)) > r:
                    pts.append((dr, dc))
    centre = pts.index((0, 0))
    pad = max(max(abs(a), abs(b)) for a, b in pts)
    return [(a + pad, b + pad) for a, b in pts], centre, pad

TOKEN_OFFSETS, CENTER_TOKEN, TOKEN_PAD = _token_offsets()
N_TOKENS = len(TOKEN_OFFSETS)               # T, no longer necessarily = P*P

N_DCT_COEFFS = config["N_DCT_COEFFS"]

USE_RAW = config["USE_RAW"]
USE_DCT = config["USE_DCT"]

MASK_INDICATOR = bool(config.get("MASK_INDICATOR", False))
FEATURE_DIM = ((N_BANDS if USE_RAW else 0) + (N_DCT_COEFFS if USE_DCT else 0) + (N_BANDS if MASK_INDICATOR else 0))    # F = C + K (+ C * mask flags)


# --------------------------- model
D_MODEL = config["D_MODEL"]
N_HEADS = config["N_HEADS"]
N_LAYERS = config["N_LAYERS"]
DROPOUT = config["DROPOUT"]

FFN_HIDDEN = config["FFN_HIDDEN"]
EMBED_HIDDEN = config["EMBED_HIDDEN"]
DECODER_HIDDEN = config["DECODER_HIDDEN"]

POS_ENCODING = config["POS_ENCODING"]

MASK_CENTER = config["MASK_CENTER"]
SPECTRAL_MASK_N = int(config.get("SPECTRAL_MASK_N", 0))
SPECTRAL_MASK_EVAL_GROUPS = int(config.get("SPECTRAL_MASK_EVAL_GROUPS", 3))
SPECTRAL_MASK_MODE = config.get("SPECTRAL_MASK_MODE", "random")
SPECTRAL_BAND_GROUPS = config.get("SPECTRAL_BAND_GROUPS") or {}

INPUT_NOISE_STD = float(config.get("INPUT_NOISE_STD", 0.0))


# --------------------------- train
EPOCHS = config["EPOCHS"]
BATCH_IMAGES = config["BATCH_IMAGES"]
CENTERS_PER_IMAGE = config["CENTERS_PER_IMAGE"]

LR = float(config["LR"])
WEIGHT_DECAY = float(config["WEIGHT_DECAY"])
WARMUP_STEPS = config["WARMUP_STEPS"]
AMP = config["AMP"]

GRAD_CLIP = config["GRAD_CLIP"]

NUM_WORKERS = config["NUM_WORKERS"]
SOURCE_BALANCE_POWER = float(config.get("SOURCE_BALANCE_POWER", 0.0))

SEED = config["SEED"]
VAL_SEED = config["VAL_SEED"]

LOSS_TOKENS = config["LOSS_TOKENS"]

EARLY_STOP_PATIENCE = config["EARLY_STOP_PATIENCE"]
LOG_EVERY = config["LOG_EVERY"]
CKPT_EVERY = config.get("CKPT_EVERY", 0)

SVDD_LAMBDA = float(config.get("SVDD_LAMBDA", 0.0))
SVDD_WARMUP_BATCHES = int(config.get("SVDD_WARMUP_BATCHES", 50))


# --------------------------- evaluation
CHUNK_PIXELS = config["CHUNK_PIXELS"]
MAX_IMAGES_PER_CLASS = config["MAX_IMAGES_PER_CLASS"]
MAX_IMAGES_PER_CLASS_OVERRIDE = config.get("MAX_IMAGES_PER_CLASS_OVERRIDE") or {}
MAX_VAL_IMAGES_PER_CLASS = config.get("MAX_VAL_IMAGES_PER_CLASS", MAX_IMAGES_PER_CLASS)

def images_per_class(cls, part="test"):
    """
        Image cap for one class and split, honouring the per-class override.
    """
    if part == "val":
        return MAX_VAL_IMAGES_PER_CLASS
    return MAX_IMAGES_PER_CLASS_OVERRIDE.get(cls, MAX_IMAGES_PER_CLASS)


MAX_NORMAL_CALIB_IMAGES = config["MAX_NORMAL_CALIB_IMAGES"]
MAX_NORMAL_TEST_IMAGES = config.get("MAX_NORMAL_TEST_IMAGES", 671)

ALPHA_GRID = config["ALPHA_GRID"]
OP_TARGET_FPR = float(config.get("OP_TARGET_FPR", 0.05))
OP_FPR_GRID = config.get("OP_FPR_GRID", [0.30, 0.20, 0.10, 0.05, 0.02, 0.01])

SELECTION_EXCLUDE = config.get("SELECTION_EXCLUDE", [])
REPORT_EXCLUDE = config.get("REPORT_EXCLUDE", [])
SAVE_MAPS_PER_CLASS = config.get("SAVE_MAPS_PER_CLASS", 0)
PCA_COMPONENTS = config.get("PCA_COMPONENTS", 4)



# ======================================================= VALIDATION
if not DATASET_PATH.is_dir():
    raise FileNotFoundError(
        f"Dataset not found at {DATASET_PATH}, "
        "if using kaggle, add:  os.environ[\"SSAD_DATA_ROOT\"] = \"/kaggle/input/sentinel2-anomaly-detection\""
    )

if PATCH_SIZE % 2 == 0:
    raise ValueError(f"PATCH_SIZE must be odd (patches are pixel-centered), got {PATCH_SIZE}")
if not (1 <= N_DCT_COEFFS <= N_BANDS):
    raise ValueError(f"N_DCT_COEFFS must be in [1, {N_BANDS}], got {N_DCT_COEFFS}")
if FEATURE_DIM == 0:
    raise ValueError(f"At least one of USE_RAW or USE_DCT must be True, got USE_RAW={USE_RAW}, USE_DCT={USE_DCT}")
if D_MODEL % N_HEADS != 0:
    raise ValueError(f"D_MODEL (= {D_MODEL}) must be divisible by N_HEADS (= {N_HEADS})")

if POS_ENCODING == "sinusoidal2d" and D_MODEL % 4 != 0:
    raise ValueError(f"D_MODEL ({D_MODEL}) must be divisible by 4 for the 2D sinusoidal encoding")
if POS_ENCODING not in ("sinusoidal2d", "learned"):
    raise ValueError(f"POS_ENCODING must be 'sinusoidal2d' or 'learned', got {POS_ENCODING!r}")

if LOSS_TOKENS not in ("all", "center"):
    raise ValueError(f"LOSS_TOKENS must be 'all' or 'center', got {LOSS_TOKENS!r}")

if not (0 <= SPECTRAL_MASK_N < N_BANDS):
    raise ValueError(f"SPECTRAL_MASK_N must be in [0, {N_BANDS}), got {SPECTRAL_MASK_N}")
if SPECTRAL_MASK_N > 0 and LOSS_TOKENS != "center":
    raise ValueError("SPECTRAL_MASK_N > 0 requires LOSS_TOKENS='center': the loss must be "
                     "computed on the hidden bands of the scored pixel only.")

if SPECTRAL_MASK_MODE not in ("random", "group"):
    raise ValueError(f"SPECTRAL_MASK_MODE must be 'random' or 'group', got {SPECTRAL_MASK_MODE!r}")
if SPECTRAL_MASK_MODE == "group":
    _seen = sorted(b for g in SPECTRAL_BAND_GROUPS.values() for b in g)
    if _seen != list(range(N_BANDS)):
        raise ValueError("SPECTRAL_BAND_GROUPS must partition all bands exactly once, "
                         f"got {_seen} for {N_BANDS} bands")