# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd

from typing import Any, Dict, List, Optional

from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .losses import FocalLoss, DiceLoss

from ct_sam.utils.metric import compute_dsc
from timm.models.layers import trunc_normal_
from monai.networks.layers.factories import Norm
from monai.networks.nets import UNet

import SimpleITK as sitk
import numpy as np


class Sam(nn.Module):
    mask_threshold: float = 0.5

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        loss_cfg: Optional[Dict] = None,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          loss_cfg: configurations for loss.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.cpp_predictor = UNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            channels=[16, 32, 64, 128, 256],
            strides=[2, 2, 2, 2],
            num_res_units=2,
            norm=Norm.INSTANCE,
        )
        
        if loss_cfg is not None:
            self.loss_func = self.get_loss_func(loss_cfg)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (_ConvNd, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    @property
    def device(self) -> Any:
        return self.image_encoder.channel_mapper1.weight.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input list(dict): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 1x1xDxHxW format,
                already transformed for input to the model.
              'mask_gt': The masks as a torch tensor in 1xNxDxHxW format,
                (# For loss computation, Optional).
              'frame': Frame information to do alignment in world coordinate.
              'original_size': (tuple(int, int, int)) The original size of
                the image before transformation, as (W, H, D) in SimpleITK format.
              'link_size': size used for mask inverse-transformation.
              'pad_flag': whether padded in preprocessing image.
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx3. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xDxHxW. (# low resolution)
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxDxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (D, H, W) is the
                original size of the image.
              'masks_itk': (List(sitk.Image)), in original size.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxDxHxW, where D=H=W=64. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        outputs = []
        for image_record in batched_input:
            if "point_coords" in image_record:
                points = (
                    image_record["point_coords"].to(self.device),
                    image_record["point_labels"].to(self.device),
                )
            else:
                points = None

            boxes = image_record.get("boxes", None)
            if boxes is not None:
                boxes = boxes.to(self.device)
            mask_inputs = image_record.get("mask_inputs", None)
            if mask_inputs is not None:
                mask_inputs = mask_inputs.to(self.device)
            curr_image_embedding = self.image_encoder(image_record["image"].to(self.device))
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_inputs,
            )

            high_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            heatmap_cross = None
            cross_supports = image_record["cross_supports"]
            cross_support_index = torch.where(cross_supports == 1)[0]
            if cross_support_index.shape[0] > 0:
                # order in channel: [image, mask, heatmap]
                images = image_record["image"][cross_support_index]
                images_crossed = images.clone()
                images_crossed[::2], images_crossed[1::2] = images[1::2], images[::2]
                cpp_inputs = torch.cat([images, (high_res_masks>self.mask_threshold)[cross_support_index].detach(), images_crossed], dim=1)
                cpp_outputs = self.cpp_predictor(cpp_inputs)
                heatmap_cross = cpp_outputs.clone()
                heatmap_cross[::2], heatmap_cross[1::2] = cpp_outputs[1::2], cpp_outputs[::2]
            
            outputs.append(
                {
                    "masks": high_res_masks,  # Tensor, heigh res logits
                    "heatmap_cross": heatmap_cross,
                    "iou_predictions": iou_predictions,
                }
            )
            
        return outputs

    def get_loss_func(self, loss_cfg):
        """Define loss function"""
        if loss_cfg.loss_name.upper() == "FOCAL":
            alpha = loss_cfg.focal_alpha
            gamma = loss_cfg.focal_gamma
            loss_func = FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_cfg.loss_name.upper() == "DICE":
            loss_func = DiceLoss()
        elif loss_cfg.loss_name.upper() == "MIX":
            alpha = loss_cfg.focal_alpha
            gamma = loss_cfg.focal_gamma
            fl = FocalLoss(alpha=alpha, gamma=gamma)
            dl = DiceLoss()
            a, b = loss_cfg.balance
            loss_func = lambda x, y: a * fl(x, y) + b * dl(x, y)  # noqa
        elif loss_cfg.loss_name.upper() == "MSE":
            loss_func = nn.MSELoss(reduction="sum")
        else:
            raise NotImplementedError(f"unspported loss function: {loss_cfg.loss_name}")

        return loss_func

    def get_loss(
        self,
        batched_pred: List[Dict[str, Any]],
        batched_true: List[Dict[str, Any]],
    ) -> torch.Tensor:
        loss_self_list = []
        loss_cross_list = []
        for pred, true in zip(batched_pred, batched_true):
            mask_gt = true["mask_gt"].to(self.device)
            heatmap_gt = true["heatmap_gt"].to(self.device)
            for p, t in zip(pred["masks"], mask_gt):
                loss_self_list.append(self.loss_func(p, t))
            if pred["heatmap_cross"] is not None:
                cross_supports = true["cross_supports"]
                cross_support_index = torch.where(cross_supports == 1)
                for p_c, t in zip(pred["heatmap_cross"], heatmap_gt[cross_support_index]):
                    loss_cross_list.append(nn.MSELoss(reduction="mean")(p_c, t))

        loss_self = torch.stack(loss_self_list).mean()
        loss_cross = loss_self.new_tensor(0)
        if len(loss_cross_list) > 0:
            loss_cross = torch.stack(loss_cross_list).mean()
        total_loss = (loss_self + loss_cross) 
        
        return total_loss, loss_self, loss_cross

    def get_metric(
        self,
        batched_pred: List[Dict[str, Any]],
        batched_true: List[Dict[str, Any]],
    ):
        dsc_list = []
        point_pool = []
        for pred, true in zip(batched_pred, batched_true):
            # Nx1xDxHxW
            fp_fn = []
            for p, t in zip(pred["masks"], true["mask_gt"].to(self.device)):
                dsc, fp, fn = compute_dsc(p.squeeze(), t.squeeze(), self.mask_threshold)
                dsc_list.append(dsc)
                fp_fn.append([fp, fn])
            point_pool.append(fp_fn)
        metric = torch.stack(dsc_list).mean()

        return metric, point_pool

    def upsample_masks(
        self,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (
                self.image_encoder.img_size,
                self.image_encoder.img_size,
                self.image_encoder.img_size,
            ),
            mode="trilinear",
            align_corners=False,
        )

        return masks

    def get_itk_masks(
        self,
        batched_pred: List[Dict[str, Any]],
        batched_true: List[Dict[str, Any]],
    ) -> List[List]:
        batched_masks_itk = []
        for pred, image_record in zip(batched_pred, batched_true):
            masks = pred["masks"]
            masks_np = (masks > self.mask_threshold).data.cpu().numpy().astype(np.uint8)
            masks_itk = []
            for i in range(masks_np.shape[0]):
                mask_np = masks_np[i, 0]
                mask_itk = sitk.GetImageFromArray(mask_np)
                masks_itk.append(mask_itk)
        batched_masks_itk.append(masks_itk)

        return batched_masks_itk
