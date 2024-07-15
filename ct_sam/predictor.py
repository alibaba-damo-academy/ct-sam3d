# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from ct_sam.models import Sam
from typing import Optional, Tuple
from .utils.itk import normalize_sitk_im


class SamPredictor:
    def __init__(
        self,
        sam_model: Sam,
        cfg,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
          cfg: data preprocessing configurations.
        """
        super().__init__()
        self.model = sam_model
        self.model.eval()
        self.reset_image()
        self.min_value = cfg.normalization_params.min_value
        self.max_value = cfg.normalization_params.max_value

    def set_image(
        self,
        image: sitk.Image,
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        img = sitk.Cast(image, sitk.sitkFloat32)
        img = normalize_sitk_im(img, self.min_value, self.max_value, True, True)
        img_tensor = (
            torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze(0).unsqueeze(0)
        ).to(self.device)

        self.set_torch_image(img_tensor)

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x1xDxHxW.
        """
        assert (
            len(transformed_image.shape) == 5
            and transformed_image.shape[1] == 1
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCDHW with target side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.input_size = tuple(transformed_image.shape[-2:])
        self.features = self.model.image_encoder(transformed_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        support_image_input: Optional[sitk.Image] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx3 array of point prompts to the
            model. Each point is in (X,Y,Z) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYZXYZ format.
          mask_input (np.ndarray): A full resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xDxHxW, where
            for SAM, e.g., D=H=W=64.
          support_image_input: A support image that can perform cross support prompt.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch, support_image_torch = None, None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch.unsqueeze(0), labels_torch.unsqueeze(0)
        if box is not None:
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch.unsqueeze(0)
        if support_image_input is not None:
            support_image_input = [F.interpolate(i, self.model.prompt_encoder.input_image_size, mode='trilinear', align_corners=False) for i in support_image_input]
            support_image_torch = torch.cat(support_image_input, dim=1)
            
            # support_image_input = sitk.Cast(support_image_input, sitk.sitkFloat32)
            # support_image_input = normalize_sitk_im(support_image_input, self.min_value, self.max_value, True, True)
            # support_image_torch = (
            #     torch.from_numpy(sitk.GetArrayFromImage(support_image_input)).unsqueeze(0).unsqueeze(0)
            # ).to(self.device)
            
        masks_itk, iou_predictions, high_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            support_image_torch,
            multimask_output,
            return_logits=return_logits,
        )

        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        high_res_masks_np = high_res_masks[0].detach().cpu().numpy()
        return masks_itk[0], iou_predictions_np, high_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        support_image: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx3 array of point prompts to the
            model. Each point is in (X,Y,Z) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx6 array given a box prompt to the
            model, in XYZXYZ format.
          mask_input (torch.Tensor or None): A full resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xDxHxW, where
            for SAM, D=H=W=64. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          support_image (torch.Tensor or None): A support image that can perform cross support prompt.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (sitk.Image): List of the output masks in CxDxHxW format, where C is the
            number of masks, and (D, H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and D=H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None
        
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        
        if support_image is not None:
            dense_embeddings = torch.cat([support_image, dense_embeddings], dim=1)
            dense_embeddings = self.model.reduce(dense_embeddings)

        # Predict masks
        high_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        
        masks_np = (high_res_masks > self.model.mask_threshold).data.cpu().numpy().astype(np.uint8)
        masks_itk = []
        for i in range(masks_np.shape[0]):
            mask_np = masks_np[i, 0]
            mask_itk = sitk.GetImageFromArray(mask_np)
            masks_itk.append(mask_itk)
        
        return masks_itk, iou_predictions, high_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxDxHxW, where C is the embedding dimension and (D,H,W) are
        the embedding spatial dimension of SAM (typically C=256, D=H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_d = None
        self.orig_h = None
        self.orig_w = None
        self.input_d = None
        self.input_h = None
        self.input_w = None
