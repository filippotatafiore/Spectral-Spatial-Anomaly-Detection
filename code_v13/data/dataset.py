import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import random

from utils.globals import DATASET_PATH, IN_CHANNELS, CLIP_REFLECTANCE, MAX_SHIFT, TEST_CLASSES, CRAT_MIN_COVER, CRAT_MAX_COVER, CRAT_GAIN_CLAMP



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


def build_donor_pool(triples, min_cover=CRAT_MIN_COVER, max_cover=CRAT_MAX_COVER):
    """
        Training anomalies usable as transplantation donors (partial masks only), grouped by anomaly type.
    """
    pool = {}
    for img_path, mask_path, cls in triples:
        cover = float((np.load(mask_path) > 0).mean())
        if min_cover <= cover <= max_cover:
            pool.setdefault(cls, []).append((img_path, mask_path))
    return pool


def _transplant(canvas, donor, dmask, gain_clamp=CRAT_GAIN_CLAMP):
    """
        Paste the masked anomaly of a donor image into a canvas (normal) image:
            - random rotation/flip of the donor (new position/orientation of the anomaly)
            - per-band gain matching the donor background to the canvas (radiometric harmonization)
            - feathered 5x5 edges (no sharp seams; the seam band is also ignored by the seg loss)
    """
    k = random.randint(0, 3)
    donor, dmask = torch.rot90(donor, k, dims=[-2, -1]), torch.rot90(dmask, k, dims=[-2, -1])
    if random.random() < 0.5:
        donor, dmask = torch.flip(donor, dims=[-1]), torch.flip(dmask, dims=[-1])

    bg = (dmask < 0.5) & (donor.abs().sum(dim=0) > 0)                       # valid donor background pixels
    if bg.sum() >= 100:  # harmonize only with enough valid background (an empty selection would give NaN gains)
        gain = (canvas.mean(dim=(1, 2)) + 1e-4) / (donor[:, bg].mean(dim=1) + 1e-4)
        donor = (donor * gain.clamp(1.0 / gain_clamp, gain_clamp).view(-1, 1, 1)).clamp(0.0, CLIP_REFLECTANCE)

    alpha = F.avg_pool2d(dmask[None, None], 5, stride=1, padding=2, count_include_pad=False)[0, 0]
    return alpha * donor + (1.0 - alpha) * canvas, dmask


class NormalImagesDataset(Dataset):
    """
        Unlabelled normal images, for unsupervised training.
    """

    def __init__(self, files, clip=CLIP_REFLECTANCE, augment=False, donors=None, crat_prob=0.0):
        if len(files) == 0:
            raise ValueError("NormalImagesDataset received an empty file list.")
        
        self.files = list(files)
        self.clip = clip
        self.augment = augment

        self.donors = donors or {}       # {anomaly type: [(img_path, mask_path), ...]} for CRAT (training only)
        self.crat_prob = crat_prob

    def __len__(self):
        return len(self.files)

    def _cross_region_transplant(self, image, source):
        """
            CRAT: transplant an anomaly whose type differs from the canvas region (types sampled uniformly -> rare types oversampled).
            Returns (image, mask, type index), or the unchanged image with type -1 if no valid transplant.
        """
        empty = torch.zeros((120, 120), dtype=torch.float32)
        types = [c for c in self.donors if c != source and self.donors[c]]
        if not types or (image.abs().sum(dim=0) == 0).float().mean() > 0.05:    # skip canvases with no-data
            return image, empty, -1
        cls = random.choice(types)
        img_path, mask_path = random.choice(self.donors[cls])
        donor = _load_image(img_path, self.clip)
        dmask = torch.from_numpy(np.load(mask_path).astype(np.uint8)).float()
        dmask = dmask * (donor.abs().sum(dim=0) > 0).float()                    # never transplant no-data pixels
        if dmask.mean() < CRAT_MIN_COVER:
            return image, empty, -1
        image, mask = _transplant(image, donor, dmask)
        return image, mask, TEST_CLASSES.index(cls)

    def __getitem__(self, index):
        image = _load_image(self.files[index], self.clip)
        mask = torch.zeros((120, 120), dtype=torch.float32)
        type_label = -1
        if self.donors and random.random() < self.crat_prob:
            image, mask, type_label = self._cross_region_transplant(image, self.files[index].parent.name)
        if self.augment:
            image, mask = _random_aug(image, mask)
        is_anom = bool(mask.max() > 0.5)
        cls_tensor = torch.tensor(1.0 if is_anom else 0.0, dtype=torch.float32)
        return image, cls_tensor, mask, torch.tensor(type_label if is_anom else -1)


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

        type_tensor = torch.tensor(TEST_CLASSES.index(class_name) if cls_label > 0 else -1)
        if self.return_meta:    # for testing
            return image, cls_tensor, mask, type_tensor, class_name, img_path.stem

        return image, cls_tensor, mask, type_tensor  # for training


def make_source_weights(files, power, mult=None):
    """
        Dataset is unbalanced -> oversample rare training sources with weight 1/n_source^power.
    """

    mult = mult or {}
    if power <= 0 and not mult:
        return None

    def _source_name(item):
        if isinstance(item, tuple):        # (img_path, mask_path, class_name) anomaly triple
            return f"{item[2]}_anomaly"
        return item.parent.name            # plain Path -> normal image

    sources = [_source_name(f) for f in files]
    counts = Counter(sources)
    return torch.tensor([mult.get(s, 1.0) / (counts[s] ** power) for s in sources], dtype=torch.double)


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