"""SAM3D factory.

Merges:
- Official MedICL-VU/PRISM public API (``sam_model_registry3D``, ``build_sam3D_vit_b_ori``)
- PRISM-placenta's ``args.use_penn`` branch which swaps in ``mask_decoder_use_penn``
  (mask conditioning via ``initial_seg_layer1/2/3`` Conv3d stack).

Set ``args.use_penn = True`` to enable mask-conditioned decoder.
"""

import torch
from functools import partial
from src.models import image_encoder, prompt_encoder, mask_decoder, sam3D, mask_decoder_use_penn


def build_sam3D_vit_b_ori(args=None, checkpoint=None):
    return _build_sam3D_ori(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        args=args,
    )


sam_model_registry3D = {
    "vit_b_ori": build_sam3D_vit_b_ori,
}


def _build_sam3D_ori(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    args=None,
):
    prompt_embed_dim = 384
    image_size = args.image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    use_penn = bool(getattr(args, "use_penn", False))
    decoder_cls = mask_decoder_use_penn.MaskDecoder3D if use_penn else mask_decoder.MaskDecoder3D

    sam = sam3D.Sam3D(
        image_encoder=image_encoder.ImageEncoderViT(
            args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=prompt_encoder.PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
            num_multiple_outputs=args.num_multiple_outputs,
            multiple_outputs=args.multiple_outputs,
        ),
        mask_decoder=decoder_cls(
            args,
            transformer_dim=prompt_embed_dim,
            num_multiple_outputs=args.num_multiple_outputs,
            multiple_outputs=args.multiple_outputs,
        ),
    )
    sam.eval()

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=args.device)

        if getattr(args, "use_sam3d_turbo", False) and getattr(args, "split", "") == "train":
            encoder_state_dict = {}
            for key in state_dict["model_state_dict"]:
                if key.startswith("image_encoder."):
                    encoder_state_dict[key[len("image_encoder."):]] = state_dict["model_state_dict"][key]
            sam.image_encoder.load_state_dict(encoder_state_dict, strict=False)
        elif use_penn:
            # PRISM-placenta: only load mask_decoder when warm-starting from a Penn-style ckpt.
            mask_decoder_state_dict = {
                k[len("mask_decoder."):]: v
                for k, v in state_dict["model_state_dict"].items()
                if k.startswith("mask_decoder.")
            }
            sam.mask_decoder.load_state_dict(mask_decoder_state_dict, strict=False)
        else:
            sam.load_state_dict(state_dict["model_state_dict"])

    # NOTE: PRISM's image_encoder is a true 3D ViT (patch_embed (768, 1, 16, 16, 16),
    # pos_embed (1, 8, 8, 8, 768), 384-channel neck). Raw SAM ViT-B (2D, 256 neck)
    # is architecturally incompatible — that's why PRISM master left --checkpoint_sam
    # dead. For warm-start, use args.use_sam3d_turbo (SAM-Med3D 3D-pretrained ckpt)
    # or pass a PRISM-trained ckpt via the ``checkpoint`` arg.
    return sam
