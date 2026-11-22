import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import random

from utils.globals import DATASET_PATH, IN_CHANNELS, CLIP_REFLECTANCE, MAX_SHIFT


def _load_image(path, clip=CLIP_REFLECTANCE):
    """
        Load (12, 120, 120) float32 image from .npy file and clip.
    """

    arr = np.load(path).astype(np.float32)
    if arr.shape[0] != IN_CHANNELS:
        raise ValueError(f"{path}: expected {IN_CHANNELS} bands, got shape {arr.shape}")

    # Clip values to [0, clip]
    if clip is not None:
        np.clip(arr, 0.0, clip, out=arr)

    return torch.from_numpy(arr)  # (12, 120, 120) = (C, H, W)


def _random_aug(image, mask):
    """
        Randomly flip, rotation, and translation of the image and mask.
    """
    # Random flip
    if random.random() < 0.5:
        image = torch.flip(image, dims=[-1])
        mask = torch.flip(mask, dims=[-1])
    if random.random() < 0.5:
        image = torch.flip(image, dims=[-2])
        mask = torch.flip(mask, dims=[-2])
    # Random rotation
    k = random.randint(0, 3)
    if k:
        image = torch.rot90(image, k, dims=[-2, -1])
        mask = torch.rot90(mask, k, dims=[-2, -1])

    # Random translation with reflect padding
    if MAX_SHIFT > 0:
        p = MAX_SHIFT
        dy, dx = random.randint(-p, p), random.randint(-p, p)
        image = F.pad(image.unsqueeze(0), (p, p, p, p), mode="reflect")[0, :, p + dy:p + dy + 120, p + dx:p + dx + 120]
        mask = F.pad(mask[None, None], (p, p, p, p), mode="reflect")[0, 0, p + dy:p + dy + 120, p + dx:p + dx + 120]
    
    return image, mask


class NormalImagesDataset(Dataset):
    """
        Unlabelled normal images, for unsupervised training.
    """

    def __init__(self, files, clip=CLIP_REFLECTANCE, augment=False):
        if len(files) == 0:
            raise ValueError("NormalImagesDataset received an empty file list.")
        
        self.files = list(files)
        self.clip = clip
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = _load_image(self.files[index], self.clip)
        mask = torch.zeros((120, 120), dtype=torch.float32)
        if self.augment:
            image, mask = _random_aug(image, mask)
        cls_tensor = torch.tensor(0.0, dtype=torch.float32)
        return image, cls_tensor, mask


class AnomalyImagesDataset(Dataset):
    """
        Anomalous images with ground-truth pixel-level binary masks.
    """

    def __init__(self, triples, clip=CLIP_REFLECTANCE, return_meta=False, augment=False):
        self.triples = list(triples)
        self.clip = clip
        self.return_meta = return_meta  # True for testing, False for training
        self.augment = augment

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        img_path, mask_path, class_name = self.triples[index]
        image = _load_image(img_path, self.clip)
        mask = torch.from_numpy(np.load(mask_path).astype(np.uint8)).float()  # (120, 120)
        if self.augment:
            image, mask = _random_aug(image, mask)
        cls_label = 1.0 if torch.max(mask) > 0.5 else 0.0  # 1 if anomalous pixels exist, else 0
        cls_tensor = torch.tensor(cls_label, dtype=torch.float32)

        if self.return_meta:    # for testing
            return image, cls_tensor, mask, class_name, img_path.stem

        return image, cls_tensor, mask  # for training


def make_source_weights(files, power):
    """
        Dataset is unbalanced -> oversample rare training sources with weight 1/n_source^power.
    """

    if power <= 0:
        return None

    def _source_name(item):
        if isinstance(item, tuple):        # (img_path, mask_path, class_name) anomaly triple
            return f"{item[2]}_anomaly"
        return item.parent.name            # plain Path -> normal image

    sources = [_source_name(f) for f in files]
    counts = Counter(sources)
    return torch.tensor([1.0 / (counts[s] ** power) for s in sources], dtype=torch.double)


def _seed_worker(worker_id):
    """
        Re-seed per DataLoader worker (otherwise forked workers
        inherit identical state, correlating augmentation choices).
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


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
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )