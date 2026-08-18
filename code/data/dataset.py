import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.globals import DATASET_PATH, N_BANDS, CLIP_REFLECTANCE


def _load_image(path, clip=CLIP_REFLECTANCE):
    """
        Load (12, 120, 120) float32 image from .npy file and clip.
    """

    arr = np.load(path).astype(np.float32)
    if arr.shape[0] != N_BANDS:
        raise ValueError(f"{path}: expected {N_BANDS} bands, got shape {arr.shape}")

    # Clip values to [0, clip]
    if clip is not None:
        np.clip(arr, 0.0, clip, out=arr)

    return torch.from_numpy(arr)  # (12, 120, 120) = (C, H, W)


class NormalImagesDataset(Dataset):
    """
        Unlabelled normal images, for unsupervised training.
    """

    def __init__(self, files, clip=CLIP_REFLECTANCE):
        if len(files) == 0:
            raise ValueError("NormalImagesDataset received an empty file list.")
        
        self.files = list(files)
        self.clip = clip

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return _load_image(self.files[index], self.clip)


class AnomalyImageDataset(Dataset):
    """
        Anomalous images with ground-truth pixel-level binary masks, for testing.
    """

    def __init__(self, triples, clip=CLIP_REFLECTANCE):
        self.triples = list(triples)
        self.clip = clip

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        img_path, mask_path, class_name = self.triples[index]
        image = _load_image(img_path, self.clip)
        mask = torch.from_numpy(np.load(mask_path).astype(np.uint8))
        return image, mask, class_name, img_path.stem


def make_loader(dataset, batch_size, num_workers=0, shuffle=False, seed=0):
    """
        Create DataLoader for the given dataset.
        Stack B (batch_size) images -> (B, C, H, W)
    """

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        generator=generator,
        persistent_workers=num_workers > 0,
    )