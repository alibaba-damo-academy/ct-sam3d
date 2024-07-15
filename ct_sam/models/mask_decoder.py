# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Union

from .spade import SPADEResnetBlock
from .common import LayerNorm3d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)  # TODO -> dsc_token
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.upscale1 = CNA(
            nn.ConvTranspose3d, transformer_dim, transformer_dim // 2, kernel_size=2, stride=2
        )
        self.reduce1 = CNA(
            nn.Conv3d, transformer_dim, transformer_dim // 8, kernel_size=3, stride=1, padding=1
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: List[torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (List[torch.Tensor]): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        # masks_hr = self.output_head(masks)
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: List[torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings[-1]
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, d, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        # Upscale mask embeddings and predict masks using the mask tokens
        x = src.transpose(1, 2).view(b, c, d, h, w)  # [1, 256, 16, 16, 16]
        x = self.upscale1(x)
        x = torch.cat([x, image_embeddings[-2]], dim=1)
        x = self.reduce1(x)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, d, h, w = x.shape

        masks = (hyper_in @ x.view(b, c, d * h * w)).view(b, -1, d, h, w)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MaskDecoderClickmap(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim

        self.num_multimask_outputs = num_multimask_outputs

        self.num_mask_tokens = num_multimask_outputs + 1
        
        self.upscale1 = CNA(nn.ConvTranspose3d, transformer_dim, transformer_dim // 2, kernel_size=2, stride=2)
        self.upscale2 = CNA(nn.ConvTranspose3d, transformer_dim, transformer_dim // 2, kernel_size=2, stride=2)
        self.upscale3 = CNA(nn.ConvTranspose3d, transformer_dim, transformer_dim // 2, kernel_size=2, stride=2)
        self.upscale4 = CNA(nn.ConvTranspose3d, transformer_dim, transformer_dim // 2, kernel_size=2, stride=2)
        self.upscale5 = CNA(nn.ConvTranspose3d, transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2)

        self.spade1 = SPADEResnetBlock(transformer_dim, transformer_dim, label_nc=3)
        self.spade2 = SPADEResnetBlock(transformer_dim, transformer_dim, label_nc=3)
        self.spade3 = SPADEResnetBlock(transformer_dim, transformer_dim, label_nc=3)
        self.spade4 = SPADEResnetBlock(transformer_dim, transformer_dim, label_nc=3)
        self.spade5 = SPADEResnetBlock(transformer_dim // 2, transformer_dim // 2, label_nc=3)

        self.mask_prediction_head = nn.Sequential(
            nn.Conv3d(transformer_dim // 4, 2, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.iou_prediction_head = MLP(
            transformer_dim // 4, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.adaptive_pooling = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(
        self,
        image_embeddings: List[torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (List[torch.Tensor]): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        # masks_hr = self.output_head(masks)
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: List[torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # SPADEResnetBlock
        prompt = torch.concat([dense_prompt_embeddings, sparse_prompt_embeddings], dim=1)
        
        x = image_embeddings[-1]
        x = self.spade1(x, prompt)
        x = self.upscale1(x) # [B, 128, 4, 4, 4]

        x = torch.cat([x, image_embeddings[-2]], dim=1)
        x = self.spade2(x, prompt)
        x = self.upscale2(x) # [B, 128, 8, 8, 8]

        x = torch.cat([x, image_embeddings[-3]], dim=1)
        x = self.spade3(x, prompt)
        x = self.upscale3(x) # [B, 128, 16, 16, 16]

        x = torch.cat([x, image_embeddings[-4]], dim=1)
        x = self.spade4(x, prompt)
        x = self.upscale4(x) # [B, 128, 32, 32, 32]

        x = self.spade5(x, prompt)
        x = self.upscale5(x) # [B, 64, 64, 64, 64]

        masks = self.mask_prediction_head(x)
        iou_pred = self.iou_prediction_head(self.adaptive_pooling(x).view(sparse_prompt_embeddings.shape[0], -1))

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class CNA(nn.Module):
    def __init__(
        self,
        conv: Union[nn.Conv3d, nn.ConvTranspose3d],
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """Basic Convolution-Normalization-Activation Block"""
        super().__init__()
        self.cna = nn.Sequential(
            conv(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            LayerNorm3d(output_dim),
            activation(),
        )

    def forward(self, x):
        return self.cna(x)
