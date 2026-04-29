"""N-click evaluation protocol ported from MedICL-VU/PRISM/src/processor/tester.py.

Given a trained model and a fold's test cases, this evaluates Dice and NSD at
varying numbers of clicks. PRISM samples random foreground voxels for each
target click count; we expose ``strategy`` so we can swap in our smarter-click
methods (uncertainty-aware, FN/FP top-1, boundary, EIG) for the paper.

Returns per-case + per-click-count metrics; caller writes them to CSV.
"""

from __future__ import annotations

import logging
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import surface_distance
from surface_distance import metrics as sd_metrics

from src.prompts.click_sim import sample_clicks
from src.utils.logger import get_logger


def _patch_dict_from_anchor(
    coords: np.ndarray, patch_size: int, vol_shape: tuple[int, int, int]
) -> dict:
    """Center a patch on the mean of the click coordinates (PRISM convention)."""
    assert coords.shape[0] >= 1
    cz, cy, cx = (
        int((coords[:, 0].max() + coords[:, 0].min()) // 2),
        int((coords[:, 1].max() + coords[:, 1].min()) // 2),
        int((coords[:, 2].max() + coords[:, 2].min()) // 2),
    )
    half = patch_size // 2
    return {
        "d_min": cz - half, "d_max": cz + half,
        "h_min": cy - half, "h_max": cy + half,
        "w_min": cx - half, "w_max": cx + half,
        "vol_shape": vol_shape,
    }


def _crop_with_pad(img: torch.Tensor, patch: dict, patch_size: int) -> tuple[torch.Tensor, dict]:
    D, H, W = patch["vol_shape"]
    d_l = max(0, -patch["d_min"]); d_r = max(0, patch["d_max"] - D)
    h_l = max(0, -patch["h_min"]); h_r = max(0, patch["h_max"] - H)
    w_l = max(0, -patch["w_min"]); w_r = max(0, patch["w_max"] - W)
    d_min = max(0, patch["d_min"]); h_min = max(0, patch["h_min"]); w_min = max(0, patch["w_min"])
    crop = img[:, :, d_min:patch["d_max"], h_min:patch["h_max"], w_min:patch["w_max"]].clone()
    crop = F.pad(crop, (w_l, w_r, h_l, h_r, d_l, d_r))
    pads = {"d_l": d_l, "d_r": d_r, "h_l": h_l, "h_r": h_r, "w_l": w_l, "w_r": w_r,
            "d_min": d_min, "h_min": h_min, "w_min": w_min}
    return crop, pads


def predict_patch(
    img_patch: torch.Tensor,
    points: torch.Tensor,
    patch_size: int,
    model_dict: dict,
    device: str,
) -> torch.Tensor:
    """Run image_encoder → prompt_encoder → mask_decoder on one cropped patch.

    Returns logits ``(B, num_classes, D, H, W)``.
    """
    img_encoder = model_dict["img_encoder"]
    prompt_encoders = model_dict["prompt_encoder_list"]
    mask_decoder = model_dict["mask_decoder"]

    out = F.interpolate(img_patch.float(), scale_factor=512 / patch_size, mode="trilinear")
    input_batch = out[0].transpose(0, 1)
    batch_features, feature_list = img_encoder(input_batch)
    feature_list.append(batch_features)

    new_feature: list[torch.Tensor] = []
    for i, (feat, pe) in enumerate(zip(feature_list, prompt_encoders)):
        if i == 3:
            new_feature.append(pe(feat.to(device), points.clone(), [patch_size, patch_size, patch_size]))
        else:
            new_feature.append(feat.to(device))
    img_resize = F.interpolate(
        img_patch[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device),
        scale_factor=64 / patch_size, mode="trilinear",
    )
    new_feature.append(img_resize)
    masks = mask_decoder(new_feature, 2, patch_size // 64)
    masks = masks.permute(0, 1, 4, 2, 3)
    return masks


def evaluate_one_case(
    img: torch.Tensor,        # (1, 1, D, H, W)
    seg: torch.Tensor,        # (1, D, H, W)
    spacing: torch.Tensor,    # (3,)
    model_dict: dict,
    *,
    patch_size: int,
    device: str,
    n_clicks: int,
    strategy: str,
    pred_prob_prev: np.ndarray | None = None,
    tolerance_mm: float = 5.0,
) -> dict:
    """Predict for one case at a single click count and return metrics."""
    seg_np = seg[0].cpu().numpy().astype(np.uint8)
    coords, _ = sample_clicks(strategy=strategy, gt=seg_np, pred_prob=pred_prob_prev, n_pos=n_clicks, n_neg=0)
    if coords.shape[0] == 0:
        return {"dice": 0.0, "nsd": 0.0, "n_clicks": n_clicks, "strategy": strategy, "skipped": True}

    # PRISM expects (x, y, z) in cropped coords; our coords are (z, y, x). Patch
    # is centered on coords; the prompt encoder normalises to [-1, 1] internally.
    vol_shape = tuple(seg_np.shape)
    patch = _patch_dict_from_anchor(coords, patch_size, vol_shape)
    img_patch, pads = _crop_with_pad(img.to(device), patch, patch_size)

    # Coords inside patch (also (z, y, x))
    local_coords = np.stack(
        [
            coords[:, 0] - patch["d_min"],
            coords[:, 1] - patch["h_min"],
            coords[:, 2] - patch["w_min"],
        ],
        axis=1,
    ).astype(np.int64)
    points = torch.from_numpy(local_coords).long().unsqueeze(0).to(device)

    logits = predict_patch(img_patch, points, patch_size, model_dict, device)
    pred_patch = F.softmax(logits, dim=1)[:, 1]
    pred_full = torch.zeros_like(img.to(device))[:, 0]
    pred_full[
        :, pads["d_min"]:patch["d_max"], pads["h_min"]:patch["h_max"], pads["w_min"]:patch["w_max"]
    ] = pred_patch[:, pads["d_l"]:patch_size - pads["d_r"], pads["h_l"]:patch_size - pads["h_r"], pads["w_l"]:patch_size - pads["w_r"]]

    pred_bin = (pred_full > 0.5).cpu().numpy()[0].astype(bool)
    gt_bin = seg_np.astype(bool)

    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    denom = float(pred_bin.sum() + gt_bin.sum())
    dice = (2 * inter / denom) if denom > 0 else 0.0

    ssd = surface_distance.compute_surface_distances(gt_bin, pred_bin, spacing_mm=spacing.cpu().numpy().tolist())
    nsd = float(sd_metrics.compute_surface_dice_at_tolerance(ssd, tolerance_mm))

    return {
        "dice": dice,
        "nsd": nsd,
        "n_clicks": n_clicks,
        "strategy": strategy,
        "skipped": False,
        "pred_prob": pred_full.detach().cpu().numpy()[0],
    }


def run_n_click_eval(
    loader: Iterable[dict],
    model_dict: dict,
    *,
    click_counts: tuple[int, ...] = (1, 3, 5, 10),
    strategy: str = "random",
    patch_size: int = 128,
    device: str = "cuda:0",
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Iterate over a dataloader and compute Dice + NSD at each ``click_counts``."""
    log = logger or get_logger("evaluator")
    rows = []
    for case_idx, batch in enumerate(loader):
        img = batch["image"]
        seg = batch["label"]
        spacing = batch["spacing"][0]
        path = batch["image_path"][0]
        for nc in click_counts:
            metrics = evaluate_one_case(
                img, seg, spacing, model_dict,
                patch_size=patch_size, device=device,
                n_clicks=nc, strategy=strategy,
            )
            row = {
                "case_path": path,
                "case_idx": case_idx,
                "n_clicks": nc,
                "strategy": strategy,
                "dice": metrics["dice"],
                "nsd": metrics["nsd"],
                "skipped": metrics["skipped"],
            }
            rows.append(row)
            log.info(
                "case %d/%d %s n_clicks=%d strategy=%s Dice=%.4f NSD=%.4f",
                case_idx, len(loader), path, nc, strategy, metrics["dice"], metrics["nsd"],
            )
    return rows
