# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .models import MaskDecoder, MaskDecoderClickmap, PromptEncoder, PromptEncoderClickmap, Sam, restv2_tiny, restv2_small, restv2_base, restv2_large
from .utils.network import partial_weight_update


def build_sam_vit_h(checkpoint=None, loss_cfg=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        loss_cfg=loss_cfg,
    )


def build_sam_vit_l(checkpoint=None, loss_cfg=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        loss_cfg=loss_cfg,
    )


def build_sam_vit_b(checkpoint=None, loss_cfg=None):
    return _build_sam(
        encoder_embed_dim=768, # default to 768
        encoder_depth=12, # default to 12
        encoder_num_heads=12, # default to 12
        encoder_global_attn_indexes=[2, 5, 8, 11], # default to [2, 5, 8, 11]
        checkpoint=checkpoint,
        loss_cfg=loss_cfg,
    )


def build_sam_restv2_tiny(checkpoint=None, loss_cfg=None):
    return _build_sam(
        encoder_embed_dim=None,
        encoder_depth=None,
        encoder_num_heads=None,
        encoder_global_attn_indexes=None,
        image_encoder=restv2_tiny(),
        checkpoint=checkpoint,
        loss_cfg=loss_cfg,
    )


def build_sam_restv2_small(checkpoint=None, loss_cfg=None):
    return _build_sam(
        encoder_embed_dim=None,
        encoder_depth=None,
        encoder_num_heads=None,
        encoder_global_attn_indexes=None,
        image_encoder=restv2_small(),
        checkpoint=checkpoint,
        loss_cfg=loss_cfg,
    )


def build_sam_restv2_base(checkpoint=None, loss_cfg=None):
    return _build_sam(
        encoder_embed_dim=None,
        encoder_depth=None,
        encoder_num_heads=None,
        encoder_global_attn_indexes=None,
        image_encoder=restv2_base(),
        checkpoint=checkpoint,
        loss_cfg=loss_cfg,
    )
    

def build_sam_restv2_large(checkpoint=None, loss_cfg=None):
    return _build_sam(
        encoder_embed_dim=None,
        encoder_depth=None,
        encoder_num_heads=None,
        encoder_global_attn_indexes=None,
        image_encoder=restv2_large(),
        checkpoint=checkpoint,
        loss_cfg=loss_cfg,
    )

def build_sam_unet_clickmap(checkpoint=None, loss_cfg=None):
    return _build_sam(
        encoder_embed_dim=None,
        encoder_depth=None,
        encoder_num_heads=None,
        encoder_global_attn_indexes=None,
        image_encoder=unet_base_tiny_encoder_only(),
        checkpoint=checkpoint,
        loss_cfg=loss_cfg,
    )

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_encoder=None,
    checkpoint=None,
    loss_cfg=None,
):
    prompt_embed_dim = 256
    image_size = 64
    vit_patch_size = 16
    image_embedding_size = image_size // 8
    
    if image_encoder is None:
        image_encoder = ImageEncoderViT(
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
            window_size=5,  # default to 14
            out_chans=prompt_embed_dim,
        )

    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoderClickmap(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderClickmap(
            num_multimask_outputs=1, # default to 3
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        loss_cfg=loss_cfg
    )

    if checkpoint is not None:
        model_state = sam.state_dict()
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
            sam = partial_weight_update(sam, state_dict["model_state_dict"])
    
    return sam


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "restv2_tiny": build_sam_restv2_tiny,
    "restv2_small": build_sam_restv2_small,
    "restv2_base": build_sam_restv2_base,
    "restv2_large": build_sam_restv2_large,
    "unet_clickmap": build_sam_unet_clickmap
}


def build_sam(cfg):
    build_sam_func = sam_model_registry[cfg.network]
    checkpoint = cfg.get("checkpoint", None)
    loss_cfg = cfg.get("loss", None)
    return build_sam_func(checkpoint=checkpoint, loss_cfg=loss_cfg)
