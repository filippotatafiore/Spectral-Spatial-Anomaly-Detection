import json

from utils.globals import SPLIT_SALT, N_DCT_COEFFS, N_BANDS, CLIP_REFLECTANCE, ARTIFACTS_DIR


# load stats
with (ARTIFACTS_DIR / "stats.json").open() as fh:
    stats = json.load(fh)

# check stats match current config
if stats["split_salt"] != SPLIT_SALT:
    raise RuntimeError("stats.json used a different SPLIT_SALT - re-run compute_stats.py.")
if stats["n_dct_coeffs"] != N_DCT_COEFFS or stats["n_bands"] != N_BANDS:
    raise RuntimeError("stats.json does not match the current config - re-run compute_stats.py.")
if stats["clip"] != CLIP_REFLECTANCE:
    raise RuntimeError("stats.json used a different CLIP_REFLECTANCE - re-run compute_stats.py.")