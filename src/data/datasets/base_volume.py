"""Modernized base 3D volume dataset, ported from PRISM ``base_dataset.py``.

Key changes from upstream PRISM:
- Dataset stats (intensity_range, mean/std, target_class, ...) are passed in
  via ``stats`` dict instead of hardcoded subclass constants. Configurable per
  dataset via YAML (R-Config.NoHardcode).
- Single channel output (no SAM RGB triple-repeat); model layer can repeat if
  required by a SAM-based backbone.
- MONAI 1.4+ API: ``EnsureChannelFirstd`` instead of deprecated ``AddChanneld``.
- Mask is binarized via ``(seg == target_class)`` followed by uint8 cast,
  matching R-Smoke.Mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    CropForegroundd,
    MapTransform,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    ScaleIntensityRanged,
    SpatialPadd,
)
from torch.utils.data import Dataset


class BinarizeLabeld(MapTransform):
    """Threshold a label map to {0, 1}; PRISM uses this for binary tumor masks (R-Smoke.Mask)."""

    def __init__(self, keys, threshold: float = 0.5, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            tensor = d[key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            d[key] = (tensor > self.threshold).to(tensor.dtype)
        return d


@dataclass
class DatasetStats:
    intensity_range: tuple[float, float]
    target_spacing: tuple[float, float, float]
    global_mean: float
    global_std: float
    spatial_index: tuple[int, int, int]
    do_dummy_2d: bool
    target_class: int


class BaseVolumeDataset(Dataset):
    """Base 3D volume dataset (CT) used by KiTS, Colon, etc."""

    def __init__(
        self,
        image_paths: Sequence[str],
        label_paths: Sequence[str],
        stats: DatasetStats,
        split: str,
        rand_crop_spatial_size: tuple[int, int, int] = (128, 128, 128),
        augmentation: bool = False,
        do_val_crop: bool = True,
    ) -> None:
        super().__init__()
        assert len(image_paths) == len(label_paths)
        self.image_paths = list(image_paths)
        self.label_paths = list(label_paths)
        self.stats = stats
        self.split = split
        self.rand_crop_spatial_size = tuple(rand_crop_spatial_size)
        self.augmentation = augmentation
        self.do_val_crop = do_val_crop
        self.transforms = self._build_transforms()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_volume(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        vol = nib.load(path)
        data = vol.get_fdata().astype(np.float32)
        zooms = np.array(vol.header.get_zooms(), dtype=np.float32)
        # Reorder to canonical (D, H, W) per spatial_index from PRISM.
        data = np.transpose(data, self.stats.spatial_index)
        zooms = zooms[list(self.stats.spatial_index)]
        return data, zooms

    def _resample_to_target(self, img: np.ndarray, seg: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = tuple(float(spacing[i] / self.stats.target_spacing[i]) for i in range(3))
        if all(abs(s - 1.0) < 1e-3 for s in scale):
            return img, seg
        img_t = torch.from_numpy(img[None, None]).float()
        seg_t = torch.from_numpy(seg[None, None]).float()
        img_r = F.interpolate(img_t, scale_factor=scale, mode="trilinear", align_corners=False)
        seg_r = F.interpolate(seg_t, scale_factor=scale, mode="trilinear", align_corners=False)
        return img_r.squeeze(0).squeeze(0).numpy(), seg_r.squeeze(0).squeeze(0).numpy()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img, spacing = self._load_volume(self.image_paths[idx])
        seg, _ = self._load_volume(self.label_paths[idx])

        img = np.nan_to_num(img, nan=0.0)
        seg = np.nan_to_num(seg, nan=0.0)

        # Binarize per R-Smoke.Mask: keep only the target class as foreground.
        seg = (seg == self.stats.target_class).astype(np.float32)

        img, seg = self._resample_to_target(img, seg, spacing)

        # MONAI dict transforms expect channel-first arrays.
        sample = {"image": img[None], "label": seg[None]}
        out = self.transforms(sample)
        # RandCropByPosNegLabeld returns a list of crops with num_samples=1.
        if isinstance(out, list):
            out = out[0]
        return {
            "image": out["image"],
            "label": out["label"].squeeze(0),
            "spacing": torch.from_numpy(spacing.astype(np.float32)),
            "image_path": self.image_paths[idx],
            "label_path": self.label_paths[idx],
        }

    def _build_transforms(self) -> Compose:
        s = self.stats
        common: list = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=s.intensity_range[0],
                a_max=s.intensity_range[1],
                b_min=s.intensity_range[0],
                b_max=s.intensity_range[1],
                clip=True,
            ),
        ]

        if self.split == "train" and self.augmentation:
            train_extra: list = [
                RandShiftIntensityd(keys=["image"], offsets=20, prob=0.5),
                CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    select_fn=lambda x: x > s.intensity_range[0],
                ),
                NormalizeIntensityd(keys=["image"], subtrahend=s.global_mean, divisor=s.global_std),
            ]
            if s.do_dummy_2d:
                train_extra.extend(
                    [
                        RandRotated(
                            keys=["image", "label"],
                            prob=0.3,
                            range_x=30 / 180 * np.pi,
                            keep_size=False,
                        ),
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.3,
                            min_zoom=[1, 0.9, 0.9],
                            max_zoom=[1, 1.1, 1.1],
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )
            else:
                train_extra.append(
                    RandZoomd(
                        keys=["image", "label"],
                        prob=0.8,
                        min_zoom=0.85,
                        max_zoom=1.25,
                        mode=["trilinear", "trilinear"],
                    )
                )
            pad_size = [round(i * 1.2) for i in self.rand_crop_spatial_size]
            train_extra.extend(
                [
                    BinarizeLabeld(keys=["label"]),
                    SpatialPadd(keys=["image", "label"], spatial_size=pad_size),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=pad_size,
                        label_key="label",
                        pos=2,
                        neg=0,
                        num_samples=1,
                    ),
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
                        random_size=False,
                    ),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                ]
            )
            return Compose(common + train_extra)

        if self.split == "val" and self.do_val_crop:
            return Compose(
                common
                + [
                    SpatialPadd(keys=["image", "label"], spatial_size=list(self.rand_crop_spatial_size)),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    NormalizeIntensityd(keys=["image"], subtrahend=s.global_mean, divisor=s.global_std),
                    BinarizeLabeld(keys=["label"]),
                ]
            )

        # split == "val" without crop, or split == "test"
        return Compose(
            common
            + [
                NormalizeIntensityd(keys=["image"], subtrahend=s.global_mean, divisor=s.global_std),
                BinarizeLabeld(keys=["label"]),
            ]
        )
