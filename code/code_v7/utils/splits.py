import hashlib
from pathlib import Path
import re

from utils.globals import (DATASET_PATH, TRAIN_SOURCES, TEST_CLASSES,
                           NORMAL_VAL_FRACTION, NORMAL_TEST_FRACTION, ANOMAL_VAL_FRACTION, ANOMAL_TEST_FRACTION,
                           SPLIT_SALT, GROUPED_SPLIT)



def stable_bucket(name, salt="", n_buckets=1000):
    """
        Randomly assign filenames into n_buckets buckets (deterministically).
        Filenames always hash to the same bucket.
    """
    digest = hashlib.sha1(f"{salt}::{name}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % n_buckets


def group_key(source, filename):
    """
        If file comes from a grouped source (ben-ge data), return the identifier of the acquisition it came from.
    """
    # ben-ge product ID is the first part of the filename (e.g. S2A_MSIL2A_20230501T103021_N0509_R108_T32TNR_20230501T123456)
    m = re.match(r"(S2[AB]_MSIL2A_\d+T\d+)_", filename)
    if m:
        return f"{source}/{m.group(1)}"
    
    return f"{source}/{filename}"  # fall back to per-file


def split_normals(root=DATASET_PATH, sources=None, val_fraction=None, test_fraction=None, salt=SPLIT_SALT):
    """
        Split normal images into (train_files, val_files, test_files), deterministically
        (data in one bucket always goes in the same split).
    """
    sources = sources or TRAIN_SOURCES
    val_fraction = NORMAL_VAL_FRACTION if val_fraction is None else val_fraction
    test_fraction = NORMAL_TEST_FRACTION if test_fraction is None else test_fraction
    val_thr = int(round(val_fraction * 1000))
    test_thr = val_thr + int(round(test_fraction * 1000))

    train_files, val_files, test_files = [], [], []
    for source in sources:
        src_dir = root / "train" / source
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing train source: {src_dir}")

        for path in sorted(src_dir.glob("*.npy")):
            # ben-ge contains multiple files from the same acquisition -> hash per acquisition (group) instead of per file
            key = group_key(source, path.name) if GROUPED_SPLIT else f"{source}/{path.name}"
            bucket = stable_bucket(key, salt)
            if bucket < val_thr:
                val_files.append(path)
            elif bucket < test_thr:
                test_files.append(path)
            else:
                train_files.append(path)

    return train_files, val_files, test_files  # file paths


def split_anomal(root=DATASET_PATH, classes=None, test_fraction=None, val_fraction=None, salt=SPLIT_SALT):
    """
        Split anomalous images

        out =
            {
                "burn_scars": {
                    "train": [ (img_path, mask_path, 'burn_scars'), (img_path, mask_path, 'burn_scars'), ... ],
                    "val": [ (img_path, mask_path, 'burn_scars'), (img_path, mask_path, 'burn_scars'), ... ],
                    "test": [ (img_path, mask_path, 'burn_scars'), (img_path, mask_path, 'burn_scars'), ... ]
                },
                "floods": {
                    "train": [ (img_path, mask_path, 'floods'), (img_path, mask_path, 'floods'), ... ],
                    "val": [ (img_path, mask_path, 'floods'), (img_path, mask_path, 'floods'), ... ],
                    "test": [ (img_path, mask_path, 'floods'), (img_path, mask_path, 'floods'), ... ]
                },
                ...
            }
    """

    classes = classes or TEST_CLASSES

    test_fraction = ANOMAL_TEST_FRACTION if test_fraction is None else test_fraction
    val_fraction = ANOMAL_VAL_FRACTION if val_fraction is None else val_fraction
    val_thr = int(round(val_fraction * 1000))
    test_thr = val_thr + int(round(test_fraction * 1000))

    out = {}
    for cls in classes:
        img_dir = root / "test" / cls / "imgs"
        mask_dir = root / "test" / cls / "masks"

        if not img_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing test class: {root / 'test' / cls}")
        
        buckets = {"train": [], "val": [], "test": []}
        for img_path in sorted(img_dir.glob("*.npy")):
            mask_path = mask_dir / f"{img_path.stem}_mask.npy"
            if not mask_path.exists():
                raise FileNotFoundError(f"No mask for {img_path}")

            bucket = stable_bucket(f"{cls}/{img_path.name}", salt)
            if bucket < val_thr:
                key = "val"
            elif bucket < test_thr:
                key = "test"
            else:
                key = "train"
            
            buckets[key].append((img_path, mask_path, cls))

        out[cls] = buckets
    return out


def _subsample_key(item):
    """
        Identity of an item: filenames only.
    """
    if isinstance(item, Path):
        return item.name
    if isinstance(item, (tuple, list)):
        return "|".join(_subsample_key(sub) for sub in item)
    return str(item)


def subsample(items, limit, salt=SPLIT_SALT):
    """
        Keep at most "limit" items by random subsampling, deterministically
        (sorts items by hash, which is consistent).
    """
    if limit is None or len(items) <= limit:
        return items
    return sorted(items, key=lambda it: stable_bucket(_subsample_key(it), salt, n_buckets=10 ** 9))[:limit]


def split_fingerprint():
    """
        Compute a fingerprint of the current split.
    """
    h = hashlib.sha256()

    train_files, val_files, test_files = split_normals()
    for tag, files in (("TRAIN", train_files), ("VAL", val_files), ("TEST", test_files)):
        for name in sorted(f"{p.parent.name}/{p.name}" for p in files):
            h.update(f"{tag}:{name}\n".encode())

    for cls, buckets in sorted(split_anomal().items()):
        for part in ("train", "val", "test"):
            for name in sorted(t[0].name for t in buckets[part]):
                h.update(f"{cls}:{part}:{name}\n".encode())

    return h.hexdigest()


if __name__ == "__main__":
    train_files, val_files, test_files = split_normals()
    print(f"normals      : {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")
    
    for cls, b in split_anomal().items():
        print(f"  {cls:<12} val={len(b['val']):5d} test={len(b['test']):5d}")
    print(f"SPLIT_SALT   : {SPLIT_SALT}")
    print(f"GROUPED_SPLIT: {GROUPED_SPLIT}")
    print(f"FINGERPRINT  : {split_fingerprint()}")