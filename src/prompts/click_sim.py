"""Click sampling strategies.

Random-FP/FN is the PRISM baseline; uncertainty-aware, FN/FP-top1 (largest
connected component centroid), boundary-aware, and EIG are our additions
(``docs/borrowed_ideas.md``). Each function returns a tensor of click coords
in (z, y, x) order with shape ``(num_points, 3)``. Positive vs negative is
encoded by accompanying labels in the prompt encoder; this module only picks
locations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return (N, 3) ndarray of (z, y, x) coordinates where mask > 0."""
    coords = np.argwhere(mask > 0)
    return coords  # shape (N, 3)


def random_in_region(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``n`` random voxels from inside a binary region (PRISM baseline)."""
    coords = _coords_from_mask(mask)
    if coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    idx = rng.choice(coords.shape[0], size=min(n, coords.shape[0]), replace=False)
    return coords[idx]


def fnfp_top1(mask: np.ndarray) -> np.ndarray:
    """Centroid of the largest connected component within ``mask``.

    Borrowed-idea-equivalent #3: PRISM samples uniformly inside FP/FN regions;
    this picks the centroid of the biggest connected component, removing the
    bias toward small noise regions.
    """
    try:
        import cc3d
    except ImportError:
        # Fallback: centroid of all foreground voxels.
        coords = _coords_from_mask(mask)
        if coords.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.int64)
        return coords.mean(axis=0, keepdims=True).astype(np.int64)
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    if labels.max() == 0:
        return np.zeros((0, 3), dtype=np.int64)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    biggest = int(sizes.argmax())
    coords = np.argwhere(labels == biggest)
    return coords.mean(axis=0, keepdims=True).astype(np.int64)


def uncertainty_topk(prob_map: np.ndarray, k: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Pick ``k`` voxels with highest binary entropy from ``prob_map`` (P(fg)).

    If ``mask`` is provided, restrict to voxels where ``mask > 0``.
    Uses analytic binary entropy and a simple top-k argpartition.
    """
    eps = 1e-6
    p = np.clip(prob_map, eps, 1 - eps)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    if mask is not None:
        entropy = np.where(mask > 0, entropy, -np.inf)
    flat = entropy.ravel()
    if not np.isfinite(flat).any():
        return np.zeros((0, 3), dtype=np.int64)
    k = min(k, int(np.isfinite(flat).sum()))
    idx = np.argpartition(flat, -k)[-k:]
    return np.array(np.unravel_index(idx, prob_map.shape), dtype=np.int64).T


def boundary_aware(prob_map: np.ndarray, k: int, threshold: float = 0.5) -> np.ndarray:
    """Pick ``k`` voxels along the predicted boundary of ``prob_map > threshold``.

    Boundary = symmetric difference of binary mask and its 1-voxel erosion.
    Among boundary voxels, rank by entropy, take top-k.
    """
    from scipy.ndimage import binary_erosion

    binary = prob_map > threshold
    eroded = binary_erosion(binary)
    boundary = binary & ~eroded
    return uncertainty_topk(prob_map, k, mask=boundary.astype(np.uint8))


def sample_clicks(
    strategy: str,
    *,
    gt: np.ndarray,
    pred_prob: Optional[np.ndarray] = None,
    n_pos: int = 1,
    n_neg: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-level dispatcher.

    Returns ``(coords, labels)`` where ``coords`` is ``(N, 3)`` int64 in
    (z, y, x) order and ``labels`` is ``(N,)`` with values in
    ``{0: negative, 1: positive}``.
    """
    rng = rng or np.random.default_rng()
    pos_region = (gt > 0).astype(np.uint8)
    neg_region = (gt == 0).astype(np.uint8)

    if strategy == "random":
        pos = random_in_region(pos_region, n_pos, rng)
        neg = random_in_region(neg_region, n_neg, rng)
    elif strategy == "fnfp_top1":
        if pred_prob is None:
            raise ValueError("fnfp_top1 requires pred_prob")
        pred_bin = (pred_prob > 0.5).astype(np.uint8)
        fn = pos_region & (1 - pred_bin)
        fp = neg_region & pred_bin
        pos = fnfp_top1(fn) if n_pos > 0 else np.zeros((0, 3), dtype=np.int64)
        neg = fnfp_top1(fp) if n_neg > 0 else np.zeros((0, 3), dtype=np.int64)
    elif strategy == "uncertainty":
        if pred_prob is None:
            raise ValueError("uncertainty requires pred_prob")
        pos = uncertainty_topk(pred_prob, n_pos, mask=pos_region) if n_pos else np.zeros((0, 3), dtype=np.int64)
        neg = uncertainty_topk(pred_prob, n_neg, mask=neg_region) if n_neg else np.zeros((0, 3), dtype=np.int64)
    elif strategy == "boundary":
        if pred_prob is None:
            raise ValueError("boundary requires pred_prob")
        pos = boundary_aware(pred_prob, n_pos) if n_pos else np.zeros((0, 3), dtype=np.int64)
        neg = boundary_aware(1 - pred_prob, n_neg) if n_neg else np.zeros((0, 3), dtype=np.int64)
    else:
        raise ValueError(f"Unknown click strategy: {strategy!r}")

    coords = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate(
        [np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)]
    )
    return coords, labels


def to_torch(coords: np.ndarray, labels: np.ndarray, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(coords).long().to(device),
        torch.from_numpy(labels).long().to(device),
    )
