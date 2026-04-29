"""Dataset registry. New datasets register their stats here.

Stats values are taken verbatim from PRISM ``dataset/datasets.py`` so
finetuning/inference numbers stay comparable to PRISM paper values
(R-Method.PrismSplit).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, DistributedSampler

from .base_volume import BaseVolumeDataset, DatasetStats


# PRISM dataset statistics (from MedICL-VU/PRISM/src/dataset/datasets.py).
DATASET_STATS: dict[str, DatasetStats] = {
    "kits21": DatasetStats(
        intensity_range=(-54, 247),
        target_spacing=(1.0, 1.0, 1.0),
        global_mean=59.53867,
        global_std=55.457336,
        spatial_index=(0, 1, 2),
        do_dummy_2d=False,
        target_class=2,
    ),
    "msd_colon": DatasetStats(
        intensity_range=(-57, 175),
        target_spacing=(1.0, 1.0, 1.0),
        global_mean=65.175035,
        global_std=32.651197,
        spatial_index=(2, 1, 0),
        do_dummy_2d=True,
        target_class=1,
    ),
    "msd_pancreas": DatasetStats(
        intensity_range=(-39, 204),
        target_spacing=(1.0, 1.0, 1.0),
        global_mean=68.45214,
        global_std=63.422806,
        spatial_index=(2, 1, 0),
        do_dummy_2d=True,
        target_class=2,
    ),
    "lits": DatasetStats(
        intensity_range=(-48, 163),
        target_spacing=(1.0, 1.0, 1.0),
        global_mean=60.057533,
        global_std=40.198017,
        spatial_index=(2, 1, 0),
        do_dummy_2d=False,
        target_class=2,
    ),
}


def load_split(split_pkl: str | Path, fold: int, split: str) -> tuple[list[str], list[str]]:
    """Load PRISM split.pkl and return (image_rel_paths, label_rel_paths) for a fold/split.

    PRISM split format: ``pickle.load(f)`` is a 5-element list (one per fold);
    each element is a dict with keys ``train``/``val``/``test``; each value is
    a dict mapping case_id -> [relative_image_path, relative_label_path].
    """
    with open(split_pkl, "rb") as f:
        folds = pickle.load(f)
    fold_dict = folds[fold][split]
    image_rels = [v[0].lstrip("/") for v in fold_dict.values()]
    label_rels = [v[1].lstrip("/") for v in fold_dict.values()]
    return image_rels, label_rels


def build_dataset(
    name: str,
    data_root: str | Path,
    split_pkl: str | Path,
    fold: int,
    split: str,
    rand_crop_spatial_size: tuple[int, int, int] = (128, 128, 128),
    augmentation: bool = False,
) -> BaseVolumeDataset:
    if name not in DATASET_STATS:
        raise KeyError(f"Unknown dataset {name!r}; known: {list(DATASET_STATS)}")
    image_rels, label_rels = load_split(split_pkl, fold, split)
    data_root = Path(data_root)
    image_paths = [str(data_root / r) for r in image_rels]
    label_paths = [str(data_root / r) for r in label_rels]
    return BaseVolumeDataset(
        image_paths=image_paths,
        label_paths=label_paths,
        stats=DATASET_STATS[name],
        split=split,
        rand_crop_spatial_size=rand_crop_spatial_size,
        augmentation=augmentation,
    )


def build_loader(
    dataset: BaseVolumeDataset,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    ddp: bool = False,
) -> DataLoader:
    """Wrap a dataset in a DataLoader. ``num_workers`` clamped to 4 (R-Process.2)."""
    if num_workers > 4:
        raise ValueError(f"num_workers={num_workers} violates R-Process.2 (max 4)")
    sampler: Any = None
    if ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=(dataset.split == "train"),
    )


__all__ = [
    "BaseVolumeDataset",
    "DatasetStats",
    "DATASET_STATS",
    "load_split",
    "build_dataset",
    "build_loader",
]
