import hashlib

from utils.globals import DATASET_PATH, TRAIN_SOURCES, TEST_CLASSES, NORMAL_VAL_FRACTION, EVAL_VAL_FRACTION, SPLIT_SALT



def stable_bucket(name, salt="", n_buckets=1000):
    """
        Randomly assign filenames into n_buckets buckets (deterministically).
        Filenames always hash to the same bucket.
    """
    digest = hashlib.sha1(f"{salt}::{name}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % n_buckets

def split_normals(root=DATASET_PATH, sources=None, val_fraction=None, salt=SPLIT_SALT):
    """
        Split training images into (train_files, val_files) determinitically
        (data in one bucket always goes in the same split).
    """

    sources = sources or TRAIN_SOURCES
    val_fraction = NORMAL_VAL_FRACTION if val_fraction is None else val_fraction
    threshold = int(round(val_fraction * 1000))

    train_files, val_files = [], []
    for source in sources:
        src_dir = root / "train" / source
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing train source: {src_dir}")
        
        for path in sorted(src_dir.glob("*.npy")):
            bucket = stable_bucket(f"{source}/{path.name}", salt)
            (val_files if bucket < threshold else train_files).append(path)

    return train_files, val_files  # file paths


def split_eval(root=DATASET_PATH, classes=None, val_fraction=None, salt=SPLIT_SALT):
    """
        Split test images into val (tuning) / test (reporting).

        out =
            {
                "burn_scars": {
                    "val": [ (img_path, mask_path, 'burn_scars'), (img_path, mask_path, 'burn_scars'), ... ],
                    "test": [ (img_path, mask_path, 'burn_scars'), (img_path, mask_path, 'burn_scars'), ... ]
                },
                "floods": {
                    "val": [ (img_path, mask_path, 'floods'), (img_path, mask_path, 'floods'), ... ],
                    "test": [ (img_path, mask_path, 'floods'), (img_path, mask_path, 'floods'), ... ]
                },
                ...
            }
    """

    classes = classes or TEST_CLASSES
    val_fraction = EVAL_VAL_FRACTION if val_fraction is None else val_fraction
    threshold = int(round(val_fraction * 1000))

    out = {}
    for cls in classes:
        img_dir = root / "test" / cls / "imgs"
        mask_dir = root / "test" / cls / "masks"

        if not img_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing test class: {root / 'test' / cls}")
        
        buckets = {"val": [], "test": []}
        for img_path in sorted(img_dir.glob("*.npy")):
            mask_path = mask_dir / f"{img_path.stem}_mask.npy"
            if not mask_path.exists():
                raise FileNotFoundError(f"No mask for {img_path}")

            bucket = stable_bucket(f"{cls}/{img_path.name}", salt)
            key = "val" if bucket < threshold else "test"
            buckets[key].append((img_path, mask_path, cls))

        out[cls] = buckets

    return out


def subsample(items, limit, salt=SPLIT_SALT):
    """
        Keep at most "limit" items by random subsampling, deteriministically (sorts items by hash, which is consistent).
    """
    if limit is None or len(items) <= limit:
        return items
    return sorted(items, key=lambda it: stable_bucket(str(it), salt, n_buckets=10 ** 9))[:limit]