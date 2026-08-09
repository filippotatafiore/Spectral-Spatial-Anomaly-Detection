from data.dataset import build_dataset

train_ds = build_dataset(split="train")
sample_img, sample_mask = train_ds[0]

print(sample_img.shape)  # (NumPatches, C, patch_size, patch_size)
print(sample_mask.shape)  # (NumPatches, patch_size, patch_size)