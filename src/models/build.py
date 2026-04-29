"""Model factory — wires PRISM image / prompt / mask modules from YAML config.

Ports the assembly logic from ``MedICL-VU/PRISM/src/config/config_setup.py``
``load_model``, with three differences:

1. Driven by our nested ``Config`` (``src/config/loader.py``) instead of argparse.
2. Optional SAM warm-start; falls back to random init when no SAM checkpoint is
   provided (PRISM's ``--no_sam`` mode).
3. ``set_trainable_params`` freezes the SAM-style image encoder except for the
   adapter modules, layer norms, and 3D-specific embeddings — same recipe as
   PRISM but extracted to its own helper so it can be tested independently.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .image_encoder import Promise
from .mask_decoder import VIT_MLAHead
from .prompt_encoder import PromptEncoder, TwoWayTransformer


def build_image_encoder(cfg) -> Promise:
    """Build the SAM-style 3D image encoder. Defaults match PRISM ViT-B."""
    enc_cfg = cfg.model.image_encoder
    return Promise(
        depth=enc_cfg.get("depth", 12),
        embed_dim=enc_cfg.get("embed_dim", 768),
        img_size=enc_cfg.get("img_size", 1024),
        mlp_ratio=enc_cfg.get("mlp_ratio", 4),
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=enc_cfg.get("num_heads", 12),
        patch_size=enc_cfg.get("patch_size", 16),
        qkv_bias=enc_cfg.get("qkv_bias", True),
        use_rel_pos=enc_cfg.get("use_rel_pos", True),
        global_attn_indexes=enc_cfg.get("global_attn_indexes", [2, 5, 8, 11]),
        window_size=enc_cfg.get("window_size", 14),
        cubic_window_size=enc_cfg.get("cubic_window_size", 8),
        out_chans=enc_cfg.get("out_chans", 256),
        num_slice=enc_cfg.get("num_slice", 16),
    )


def build_prompt_encoders(cfg, n: int = 4) -> nn.ModuleList:
    """Build ``n`` parallel prompt encoders (PRISM uses 4, one per neck level)."""
    encoders = nn.ModuleList()
    for _ in range(n):
        encoders.append(
            PromptEncoder(
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                )
            )
        )
    return encoders


def build_mask_decoder(cfg) -> VIT_MLAHead:
    dec_cfg = cfg.model.mask_decoder
    return VIT_MLAHead(
        img_size=dec_cfg.get("img_size", 96),
        num_classes=dec_cfg.get("num_classes", 2),
    )


def set_trainable_params(image_encoder: Promise) -> None:
    """Freeze image encoder backbone; unfreeze 3D-specific bits + adapters.

    Matches PRISM's recipe: depth_embed, slice_embed, every block's norm1/2 +
    adapter + adapter_back + a learned rel_pos_d, and the neck_3d projection.
    """
    for p in image_encoder.parameters():
        p.requires_grad = False
    image_encoder.depth_embed.requires_grad = True
    for p in image_encoder.slice_embed.parameters():
        p.requires_grad = True
    for blk in image_encoder.blocks:
        for p in blk.norm1.parameters():
            p.requires_grad = True
        for p in blk.adapter.parameters():
            p.requires_grad = True
        for p in blk.adapter_back.parameters():
            p.requires_grad = True
        for p in blk.norm2.parameters():
            p.requires_grad = True
        # PRISM initializes rel_pos_d as the average of rel_pos_h/w then trains it.
        blk.attn.rel_pos_d = nn.Parameter(
            0.5 * (blk.attn.rel_pos_h + blk.attn.rel_pos_w),
            requires_grad=True,
        )
    for layer in image_encoder.neck_3d:
        for p in layer.parameters():
            p.requires_grad = True


def maybe_load_sam_weights(image_encoder: Promise, sam_ckpt: Optional[str]) -> bool:
    """Optionally load SAM ViT-B weights into the image encoder. Returns True if loaded."""
    if not sam_ckpt:
        return False
    sam_ckpt_path = Path(sam_ckpt)
    if not sam_ckpt_path.exists():
        return False
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError as exc:
        raise ImportError(
            "segment-anything not installed. Run via cpu.slurm: "
            "`pip install segment-anything`"
        ) from exc
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt_path))
    mask_generator = SamAutomaticMaskGenerator(sam)
    image_encoder.load_state_dict(
        mask_generator.predictor.model.image_encoder.state_dict(),
        strict=False,
    )
    del sam, mask_generator
    return True


def build_model(cfg, device: str = "cuda:0") -> dict:
    """One-stop factory; returns ``{img_encoder, prompt_encoder_list, mask_decoder}``."""
    img_encoder = build_image_encoder(cfg)
    sam_path = cfg.model.image_encoder.get("sam_checkpoint", None) if hasattr(cfg.model.image_encoder, "get") else cfg.model.image_encoder.get("sam_checkpoint")
    loaded_sam = maybe_load_sam_weights(img_encoder, sam_path)
    set_trainable_params(img_encoder)
    img_encoder.to(device)

    prompt_encoder_list = build_prompt_encoders(cfg, n=cfg.model.get("n_prompt_encoders", 4))
    prompt_encoder_list.to(device)

    mask_decoder = build_mask_decoder(cfg)
    mask_decoder.to(device)

    return {
        "img_encoder": img_encoder,
        "prompt_encoder_list": prompt_encoder_list,
        "mask_decoder": mask_decoder,
        "loaded_sam": loaded_sam,
    }


def count_trainable_params(model_dict: dict) -> dict[str, int]:
    """Diagnostic: per-component trainable parameter counts."""
    counts = {}
    for name in ("img_encoder", "mask_decoder"):
        m = model_dict[name]
        counts[name] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    counts["prompt_encoders"] = sum(
        p.numel() for m in model_dict["prompt_encoder_list"] for p in m.parameters() if p.requires_grad
    )
    counts["total"] = sum(counts.values())
    return counts
