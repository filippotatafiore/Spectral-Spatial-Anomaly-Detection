import json

from utils.globals import ARTIFACTS_DIR, CLIP_REFLECTANCE, N_BANDS, N_DCT_COEFFS, SPLIT_SALT, TRAIN_SOURCES


def load_and_check_stats(path=None):
    """
        Load stats.json and verify it matches the current config.
    """

    path = path or (ARTIFACTS_DIR / "stats.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python compute_stats.py` first.")

    with path.open() as fh:
        stats = json.load(fh)

    checks = [
        ("split_salt", stats.get("split_salt"), SPLIT_SALT),
        ("n_bands", stats.get("n_bands"), N_BANDS),
        ("n_dct_coeffs", stats.get("n_dct_coeffs"), N_DCT_COEFFS),
        ("clip", stats.get("clip"), CLIP_REFLECTANCE),
        ("train_sources", list(stats.get("train_sources", [])), list(TRAIN_SOURCES)),
    ]
    for name, got, want in checks:
        if got != want:
            raise RuntimeError(
                f"stats.json does not match current config: {name} = {got!r}, but config has = {want!r}. "
                f"Re-run python compute_stats.py ."
            )

    expected = N_BANDS + N_DCT_COEFFS
    if len(stats["feature_mean"]) != expected or len(stats["feature_std"]) != expected:
        raise RuntimeError(f"stats.json feature vectors are not length {expected}.")

    return stats