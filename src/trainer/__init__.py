"""Single-GPU training loop ported from MedICL-VU/PRISM/src/processor/trainer.py.

Differences from upstream PRISM:
- Driven by our YAML config (``src/config/loader.py``) instead of argparse.
- ``mp.spawn`` DDP removed for v1; switch to ``torchrun`` later if needed.
- Uses our dataset interface (``src/data/datasets.build_loader``) which yields a
  dict with ``image``/``label``/``spacing`` keys, not the legacy 3-tuple.
- Click sampling routed through ``src/prompts/click_sim.sample_clicks`` so the
  smarter-click ablations (uncertainty-aware, FN/FP top-1, etc.) can plug in.

Boundary distance loss is omitted in v1; add as ``--loss.boundary_weight``
ablation after smoke-training succeeds.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceLoss
from torch.optim import AdamW, lr_scheduler

from src.data.datasets import build_dataset, build_loader
from src.models.build import build_model, count_trainable_params
from src.prompts.click_sim import sample_clicks
from src.utils.logger import get_logger
from src.utils.seed import set_all_seeds


def _save_checkpoint(state: dict, is_best: bool, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last = ckpt_dir / "last.pth.tar"
    torch.save(state, last)
    if is_best:
        shutil.copyfile(last, ckpt_dir / "best.pth.tar")


def _sample_train_clicks(
    seg: torch.Tensor,
    n_pos: int,
    n_neg: int,
    device: str,
) -> torch.Tensor:
    """Random FP/FN-style click sampling (PRISM baseline).

    Args:
        seg: ``(B, D, H, W)`` binary segmentation tensor.
    Returns:
        ``(1, N, 3)`` long tensor of click coordinates in (D, H, W) order, on
        ``device``. Returns positive + negative concatenated.
    """
    seg_np = seg[0].detach().cpu().numpy().astype(np.uint8)
    coords, _ = sample_clicks(strategy="random", gt=seg_np, n_pos=n_pos, n_neg=n_neg)
    if coords.shape[0] == 0:
        # No foreground: sample only negatives
        bg_coords, _ = sample_clicks(
            strategy="random",
            gt=1 - seg_np,
            n_pos=n_neg,
            n_neg=0,
        )
        coords = bg_coords
    # Float — grid_sample inside the prompt encoder requires Float coords
    # (PRISM does .float() in src/utils/util.py:get_points).
    return torch.from_numpy(coords).float().unsqueeze(0).to(device)


def _model_forward(
    img: torch.Tensor,
    seg: torch.Tensor,
    model_dict: dict,
    patch_size: int,
    device: str,
    n_pos: int,
    n_neg: int,
) -> torch.Tensor:
    """One forward pass through PRISM image_encoder → prompt_encoder → mask_decoder.

    Returns logits with shape ``(B, num_classes, D, H, W)``.
    """
    img_encoder = model_dict["img_encoder"]
    prompt_encoders = model_dict["prompt_encoder_list"]
    mask_decoder = model_dict["mask_decoder"]

    out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode="trilinear")
    input_batch = out.to(device)
    if img.shape[0] == 1:
        input_batch = input_batch[0].transpose(0, 1)
    else:
        input_batch = input_batch.transpose(0, 1)

    batch_features, feature_list = img_encoder(input_batch)
    feature_list.append(batch_features)

    points = _sample_train_clicks(seg, n_pos=n_pos, n_neg=n_neg, device=device)

    new_feature: list[torch.Tensor] = []
    for i, (feat, pe) in enumerate(zip(feature_list, prompt_encoders)):
        if i == 3:
            new_feature.append(pe(feat, points.clone(), [patch_size, patch_size, patch_size]))
        else:
            new_feature.append(feat)
    img_resize = F.interpolate(
        img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device),
        scale_factor=64 / patch_size,
        mode="trilinear",
    )
    new_feature.append(img_resize)
    masks = mask_decoder(new_feature, 2, patch_size // 64)
    masks = masks.permute(0, 1, 4, 2, 3)  # → (B, num_classes, D, H, W)
    return masks


class Trainer:
    """One-GPU training driver. ``cfg`` is a nested ``Config`` from YAML."""

    def __init__(self, cfg, logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger or get_logger("trainer")
        self.device = cfg.train.get("device", "cuda:0") if hasattr(cfg.train, "get") else "cuda:0"
        set_all_seeds(cfg.experiment.seed)

        self.output_dir = Path(cfg.experiment.output_dir)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"

        # Data
        self._build_data()
        # Model
        self.model_dict = build_model(cfg, device=self.device)
        self.logger.info("trainable params: %s", count_trainable_params(self.model_dict))
        # Loss / optim / scheduler
        self._build_loss()
        self._build_optimizer()
        self._build_scheduler()

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def _build_data(self) -> None:
        d = self.cfg.data
        train_ds = build_dataset(
            name=d.dataset,
            data_root=d.data_root + "/" + d.dataset,
            split_pkl=d.split_pkl,
            fold=d.fold,
            split="train",
            rand_crop_spatial_size=tuple(d.patch_size),
            augmentation=True,
        )
        val_ds = build_dataset(
            name=d.dataset,
            data_root=d.data_root + "/" + d.dataset,
            split_pkl=d.split_pkl,
            fold=d.fold,
            split="val",
            rand_crop_spatial_size=tuple(d.patch_size),
            augmentation=False,
        )
        self.train_loader = build_loader(
            train_ds, batch_size=self.cfg.train.batch_size, num_workers=d.num_workers, shuffle=True
        )
        self.val_loader = build_loader(
            val_ds, batch_size=1, num_workers=d.num_workers, shuffle=False
        )
        self.logger.info("train cases=%d val cases=%d", len(train_ds), len(val_ds))

    def _build_loss(self) -> None:
        self.train_loss = DiceCELoss(
            include_background=False, softmax=True, to_onehot_y=True,
            lambda_dice=0.5, lambda_ce=0.5,
        )
        self.val_loss = DiceLoss(
            include_background=False, softmax=True, to_onehot_y=True, reduction="none"
        )

    def _build_optimizer(self) -> None:
        lr = self.cfg.train.lr
        wd = self.cfg.train.weight_decay
        self.opt_encoder = AdamW(
            [p for p in self.model_dict["img_encoder"].parameters() if p.requires_grad],
            lr=lr, weight_decay=wd,
        )
        prompt_params = [
            p for m in self.model_dict["prompt_encoder_list"] for p in m.parameters() if p.requires_grad
        ]
        self.opt_prompt = AdamW(prompt_params, lr=lr, weight_decay=wd)
        self.opt_decoder = AdamW(
            [p for p in self.model_dict["mask_decoder"].parameters() if p.requires_grad],
            lr=lr, weight_decay=wd,
        )

    def _build_scheduler(self) -> None:
        self.sched_encoder = lr_scheduler.LinearLR(self.opt_encoder, start_factor=1.0, end_factor=0.01, total_iters=500)
        self.sched_prompt = lr_scheduler.LinearLR(self.opt_prompt, start_factor=1.0, end_factor=0.01, total_iters=500)
        self.sched_decoder = lr_scheduler.LinearLR(self.opt_decoder, start_factor=1.0, end_factor=0.01, total_iters=500)

    def resume(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.start_epoch = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.model_dict["img_encoder"].load_state_dict(ckpt["encoder_dict"], strict=True)
        for i, pe in enumerate(self.model_dict["prompt_encoder_list"]):
            pe.load_state_dict(ckpt["prompt_dict"][i], strict=True)
        self.model_dict["mask_decoder"].load_state_dict(ckpt["decoder_dict"], strict=True)
        self.opt_encoder.load_state_dict(ckpt["encoder_opt"])
        self.opt_prompt.load_state_dict(ckpt["prompt_opt"])
        self.opt_decoder.load_state_dict(ckpt["decoder_opt"])
        self.sched_encoder.load_state_dict(ckpt["encoder_scheduler"])
        self.sched_prompt.load_state_dict(ckpt["prompt_scheduler"])
        self.sched_decoder.load_state_dict(ckpt["decoder_scheduler"])
        self.logger.info("Resume from epoch %d (best_val_loss=%.4f)", self.start_epoch, self.best_val_loss)

    def run(self) -> None:
        max_epoch = self.cfg.train.epochs
        patch_size = self.cfg.data.patch_size[0]
        for epoch in range(self.start_epoch, max_epoch):
            self._train_one_epoch(epoch, patch_size)
            val_loss = self._validate(epoch, patch_size)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            self._save(epoch, is_best)
            self.logger.info("epoch %d done — val_loss=%.4f best=%.4f@%d",
                             epoch, val_loss, self.best_val_loss, self.best_epoch)

    def _train_one_epoch(self, epoch: int, patch_size: int) -> None:
        for m in self.model_dict["prompt_encoder_list"]:
            m.train()
        self.model_dict["img_encoder"].train()
        self.model_dict["mask_decoder"].train()

        losses = []
        t0 = time.time()
        max_steps = self.cfg.train.get("max_steps_per_epoch", 0) if hasattr(self.cfg.train, "get") else 0
        for idx, batch in enumerate(self.train_loader):
            if max_steps and idx >= max_steps:
                break
            img, seg = batch["image"], batch["label"]
            masks = _model_forward(
                img, seg, self.model_dict, patch_size, self.device,
                n_pos=self.cfg.click.get("n_pos_train", 50) if hasattr(self.cfg.click, "get") else 50,
                n_neg=self.cfg.click.get("n_neg_train", 50) if hasattr(self.cfg.click, "get") else 50,
            )
            seg_d = seg.to(self.device).unsqueeze(1)  # (B, 1, D, H, W) for DiceCE to_onehot_y
            loss = self.train_loss(masks, seg_d)

            self.opt_encoder.zero_grad()
            self.opt_prompt.zero_grad()
            self.opt_decoder.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_dict["img_encoder"].parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model_dict["mask_decoder"].parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model_dict["prompt_encoder_list"][-1].parameters(), 1.0)
            self.opt_encoder.step()
            self.opt_prompt.step()
            self.opt_decoder.step()

            losses.append(float(loss.detach().cpu().numpy()))
            if idx % self.cfg.log.every_n_steps == 0:
                self.logger.info("epoch %d iter %d/%d loss=%.4f", epoch, idx, len(self.train_loader), losses[-1])

        self.sched_encoder.step()
        self.sched_prompt.step()
        self.sched_decoder.step()
        self.logger.info("epoch %d train_loss=%.4f time=%.1fs", epoch, float(np.mean(losses)), time.time() - t0)

    @torch.no_grad()
    def _validate(self, epoch: int, patch_size: int) -> float:
        for m in self.model_dict["prompt_encoder_list"]:
            m.eval()
        self.model_dict["img_encoder"].eval()
        self.model_dict["mask_decoder"].eval()

        losses = []
        for batch in self.val_loader:
            img, seg = batch["image"], batch["label"]
            masks = _model_forward(
                img, seg, self.model_dict, patch_size, self.device,
                n_pos=self.cfg.click.get("n_pos_val", 10) if hasattr(self.cfg.click, "get") else 10,
                n_neg=self.cfg.click.get("n_neg_val", 10) if hasattr(self.cfg.click, "get") else 10,
            )
            seg_d = seg.to(self.device).unsqueeze(1)
            loss = self.val_loss(masks, seg_d)
            losses.append(float(loss.detach().cpu().numpy()))
        return float(np.mean(losses))

    def _save(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch + 1,
            "best_val_loss": self.best_val_loss,
            "encoder_dict": self.model_dict["img_encoder"].state_dict(),
            "decoder_dict": self.model_dict["mask_decoder"].state_dict(),
            "prompt_dict": [m.state_dict() for m in self.model_dict["prompt_encoder_list"]],
            "encoder_opt": self.opt_encoder.state_dict(),
            "prompt_opt": self.opt_prompt.state_dict(),
            "decoder_opt": self.opt_decoder.state_dict(),
            "encoder_scheduler": self.sched_encoder.state_dict(),
            "prompt_scheduler": self.sched_prompt.state_dict(),
            "decoder_scheduler": self.sched_decoder.state_dict(),
        }
        _save_checkpoint(state, is_best, self.ckpt_dir)
