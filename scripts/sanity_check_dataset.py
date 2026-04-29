"""Sanity-check the data layer end-to-end on a single case.

This script is the smallest D1 smoke test (R-Smoke.1): it verifies the data
pipeline before any model code exists. Run after the lab->ACCRE rsync drops
its first KiTS21 case onto ACCRE.

Usage:
    python scripts/sanity_check_dataset.py --config configs/exp/smoke/config.yaml
or with no config (defaults to KiTS21 + first val case):
    python scripts/sanity_check_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataset, build_loader
from src.prompts.click_sim import sample_clicks
from src.utils.logger import get_logger
from src.utils.seed import set_all_seeds


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="kits21", choices=["kits21", "msd_colon", "msd_pancreas", "lits"])
    p.add_argument("--data_root", default="/data/h_oguz_lab/lih30/interactive_seg_data")
    p.add_argument("--split_pkl", default=None, help="defaults to splits/<dataset>.pkl")
    p.add_argument("--split", default="val")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--patch", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--n", type=int, default=1, help="number of cases to inspect")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_all_seeds(args.seed)
    log = get_logger("sanity")

    if args.split_pkl is None:
        # splits/ in the repo holds kits.pkl and colon.pkl; map dataset names accordingly.
        short = {"kits21": "kits", "msd_colon": "colon", "msd_pancreas": "pancreas", "lits": "lits"}[args.dataset]
        args.split_pkl = str(ROOT / "splits" / f"{short}.pkl")

    data_root = Path(args.data_root) / args.dataset
    log.info("data_root=%s split_pkl=%s split=%s fold=%d", data_root, args.split_pkl, args.split, args.fold)

    ds = build_dataset(
        name=args.dataset,
        data_root=data_root,
        split_pkl=args.split_pkl,
        fold=args.fold,
        split=args.split,
        rand_crop_spatial_size=tuple(args.patch),
        augmentation=False,
    )
    log.info("dataset built, %d cases in fold %d / %s", len(ds), args.fold, args.split)

    # Filter to cases that actually exist on disk (sync may be incomplete).
    keep_idx = [i for i, p in enumerate(ds.image_paths) if Path(p).exists()]
    log.info("cases present on disk: %d / %d", len(keep_idx), len(ds))
    if not keep_idx:
        log.error("no cases found on disk under %s; sync may not have started", data_root)
        return 2
    ds.image_paths = [ds.image_paths[i] for i in keep_idx]
    ds.label_paths = [ds.label_paths[i] for i in keep_idx]

    loader = build_loader(ds, batch_size=1, num_workers=2, shuffle=False)

    n_seen = 0
    for batch in loader:
        img = batch["image"]
        seg = batch["label"]
        spacing = batch["spacing"]
        path = batch["image_path"][0]
        log.info("--- case %s ---", path)
        log.info("img shape=%s dtype=%s min=%.2f max=%.2f mean=%.2f",
                 tuple(img.shape), img.dtype, float(img.min()), float(img.max()), float(img.mean()))
        log.info("seg shape=%s dtype=%s unique=%s sum=%d",
                 tuple(seg.shape), seg.dtype, torch.unique(seg).tolist(), int(seg.sum()))
        log.info("spacing (mm)=%s", spacing.squeeze(0).tolist())

        # R-Smoke checks (numbers, not just "didn't crash").
        assert seg.max() <= 1.0 and seg.min() >= 0.0, "seg out of [0,1] — mask 0/255 bug"
        assert torch.isfinite(img).all() and torch.isfinite(seg).all(), "NaN/Inf in tensors"
        assert img.shape[-3:] == seg.shape[-3:], "image/label spatial shape mismatch"

        # Click sampling sanity: pick 1 positive click (random) and 1 fnfp_top1 click.
        seg_np = seg.squeeze().cpu().numpy()
        if seg_np.sum() > 0:
            random_coords, _ = sample_clicks("random", gt=seg_np, n_pos=1, n_neg=0)
            fake_pred = np.full_like(seg_np, 0.3, dtype=np.float32)  # pretend model output
            top1_coords, _ = sample_clicks("fnfp_top1", gt=seg_np, pred_prob=fake_pred, n_pos=1, n_neg=0)
            log.info("random click=%s fnfp_top1 click=%s",
                     random_coords.tolist(), top1_coords.tolist())
        else:
            log.warning("seg has no foreground voxels in this crop; click sampler skipped")

        n_seen += 1
        if n_seen >= args.n:
            break

    log.info("sanity check passed: %d case(s) loaded and validated", n_seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
