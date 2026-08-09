import numpy as np
import torch
from torch.utils.data import Dataset

from pathlib import Path

from data.transforms import PatchExtractor, SpectralDCT
from utils.globals import DATASET_PATH, BANDS



def _to_channel_first(x):
    """
    Convert x to channel-first tensor (C, H, W) and float32.
    """

    # convert to torch tensor and float32
    if isinstance(x, torch.Tensor):
        tensor = x.float()
    else:
        tensor = torch.from_numpy(np.asarray(x)).float()

    # convert to channel-first format
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[-1] == len(BANDS) and tensor.shape[0] != len(BANDS):
            tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(tensor.shape)}")

    return tensor


class PatchDCTTransform(torch.nn.Module):
    """
        Transform input image tensor: extract patches + DCT along spectral dimension.
    """

    def __init__(self, patch_size=9, stride=None, scale_factor=None, num_bands=None):
        super().__init__()
        self.patch_extractor = PatchExtractor(patch_size=patch_size, stride=stride, scale_factor=scale_factor)
        n_bands = num_bands if num_bands is not None else len(BANDS)
        self.spectral_dct = SpectralDCT(num_bands=n_bands)

    def forward(self, img_tensor):
        patches = self.patch_extractor(img_tensor)
        return self.spectral_dct(patches)  # (Num_Patches, C, patch_H, patch_W)


class Sentinel2Dataset(Dataset):
    """
        Custom Dataset for Sentinel-2 anomaly detection.
    """

    def __init__(self, split="train", transform=None):
        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test'].")

        self.split = split
        self.split_dir = DATASET_PATH / split
        self.transform = transform

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.split_dir}")

        self.samples = self._extract_samples(self)

    def _extract_samples(self):
        # Extract samples from the dataset    
        self.samples = []
        for dir in sorted(self.split_dir.iterdir()):
            if self.split == "train":
                for np_file in sorted(dir.glob("*.npy")):
                    self.samples.append((np_file, None))
            else:
                imgs_dir = dir / "imgs"
                masks_dir = dir / "masks"
                for np_file in sorted(imgs_dir.glob("*.npy")):
                    img_file_name = np_file.stem
                    mask_file_name = img_file_name + "_mask.npy"
                    mask_np_file = masks_dir / mask_file_name

                    if not mask_np_file.exists():
                        raise FileNotFoundError(f"Mask file not found for {np_file}: {mask_np_file}")

                    self.samples.append((np_file, mask_np_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        np_file, mask_np_file = self.samples[index]

        image = np.load(np_file)
        image = _to_channel_first(image)

        # apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, mask_np_file



def build_dataset(split="train", patch_size=9, stride=None, scale_factor=None, num_bands=None):
    transform = PatchDCTTransform(patch_size=patch_size, stride=stride, scale_factor=scale_factor, num_bands=num_bands)
    return Sentinel2Dataset(split=split, transform=transform)