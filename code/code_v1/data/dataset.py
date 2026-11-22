import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.utils.data import WeightedRandomSampler

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


class AnomalyImagesDataset(Dataset):
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


def make_source_weights(files, power):
    """
        Dataset is unbalanced -> oversample rare training sources with weight 1/n_source^power.
    """

    if power <= 0:
        return None

    sources = [p.parent.name for p in files]
    counts = Counter(sources)
    return torch.tensor([1.0 / (counts[s] ** power) for s in sources], dtype=torch.double)


def make_loader(dataset, batch_size, num_workers=0, shuffle=False, seed=0, weights=None):
    """
        Create DataLoader for the given dataset.
        Stack B (batch_size) images -> (B, C, H, W)

        weights: per-sample sampling weights computed with make_source_weights().
    """

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    # WeightedRandomSampler replaces shuffling.
    sampler = None
    if weights is not None and shuffle:
        sampler = WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=True, generator=generator
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=sampler is not None or shuffle,
        generator=generator,
        persistent_workers=num_workers > 0,
    )