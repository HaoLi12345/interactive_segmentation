"""Entry point: run n-click evaluation on a held-out test split.

Backends:
    --backend prism         load a PRISM checkpoint into our build_model()
    --backend ours          load a finetuned checkpoint trained by our trainer
    (vista3d / nninteractive backends to be added in TASK-009)

Output: CSV at ``<experiment.output_dir>/metrics/per_case.csv``.

Run only via SLURM (R-Process.LoginNode).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.loader import load_yaml, apply_overrides
from src.data.datasets import build_dataset, build_loader
from src.evaluator import run_n_click_eval
from src.models.build import build_model
from src.utils.logger import get_logger
from src.utils.seed import set_all_seeds


def _load_prism_checkpoint(model_dict: dict, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_dict["img_encoder"].load_state_dict(ckpt["encoder_dict"], strict=True)
    for i, pe in enumerate(model_dict["prompt_encoder_list"]):
        pe.load_state_dict(ckpt["prompt_dict"][i], strict=True)
    model_dict["mask_decoder"].load_state_dict(ckpt["decoder_dict"], strict=True)
    model_dict["img_encoder"].eval()
    for pe in model_dict["prompt_encoder_list"]:
        pe.eval()
    model_dict["mask_decoder"].eval()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--backend", choices=["prism", "ours"], default="prism")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (PRISM or ours)")
    p.add_argument("--strategy", default="random",
                   choices=["random", "fnfp_top1", "uncertainty", "boundary"])
    p.add_argument("--clicks", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--max_cases", type=int, default=None,
                   help="Limit cases for smoke runs")
    p.add_argument("--split", default="test")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.set)

    set_all_seeds(cfg.experiment.seed)
    log = get_logger("eval", log_file=Path(cfg.experiment.output_dir) / "logs" / "eval.log")
    log.info("backend=%s ckpt=%s strategy=%s clicks=%s split=%s",
             args.backend, args.ckpt, args.strategy, args.clicks, args.split)

    d = cfg.data
    test_ds = build_dataset(
        name=d.dataset,
        data_root=str(Path(d.data_root) / d.dataset),
        split_pkl=d.split_pkl,
        fold=d.fold,
        split=args.split,
        rand_crop_spatial_size=tuple(d.patch_size),
        augmentation=False,
    )
    if args.max_cases is not None:
        test_ds.image_paths = test_ds.image_paths[: args.max_cases]
        test_ds.label_paths = test_ds.label_paths[: args.max_cases]
    loader = build_loader(test_ds, batch_size=1, num_workers=2, shuffle=False)
    log.info("eval cases=%d", len(test_ds))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dict = build_model(cfg, device=device)
    _load_prism_checkpoint(model_dict, args.ckpt)

    rows = run_n_click_eval(
        loader=loader,
        model_dict=model_dict,
        click_counts=tuple(args.clicks),
        strategy=args.strategy,
        patch_size=cfg.data.patch_size[0],
        device=device,
        logger=log,
    )

    out_csv = Path(cfg.experiment.output_dir) / "metrics" / f"eval_{args.backend}_{args.strategy}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["case_path", "n_clicks", "strategy", "dice", "nsd", "skipped"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    log.info("metrics written: %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
