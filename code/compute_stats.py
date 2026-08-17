import json

from utils.splits import split_normals
from data.stats import compute_stats

from utils.globals import N_BANDS, N_DCT_COEFFS, CLIP_REFLECTANCE, SPLIT_SALT, TRAIN_SOURCES, ARTIFACTS_DIR

def main():
    # Split training images into (train_files, val_files)
    train_files, val_files = split_normals()
    print(f"Normal images: {len(train_files)} train / {len(val_files)} val")

    # Compute statistics over train split
    stats = compute_stats(train_files, N_BANDS, N_DCT_COEFFS, clip=CLIP_REFLECTANCE)

    stats["split_salt"] = SPLIT_SALT
    stats["train_sources"] = TRAIN_SOURCES

    out_path = ARTIFACTS_DIR / "stats.json"
    with out_path.open("w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nWrote {out_path}")
    print(f"\tpixels       : {stats['n_pixels']:,}")
    print(f"\tfeature_mean : {[round(v, 4) for v in stats['feature_mean']]}")
    print(f"\tfeature_std  : {[round(v, 4) for v in stats['feature_std']]}")


if __name__ == "__main__":
    main()