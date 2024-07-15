# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm3d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int, int)): The spatial size of the
            image embedding, as (D, H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (D, H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
            4 * image_embedding_size[2],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv3d(1, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans),
            activation(),
            # nn.Conv3d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            # LayerNorm3d(mask_in_chans),
            # activation(),
            nn.Conv3d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 3), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 3)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_D)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            print("coords and labels shape: ", coords.shape, labels.shape)
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                bs,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
                self.image_embedding_size[2],
            )

        return sparse_embeddings, dense_embeddings


class PromptEncoderClickmap(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int, int)): The spatial size of the
            image embedding, as (D, H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (D, H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
            4 * image_embedding_size[2],
        )

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        # return self.pe_layer(self.image_embedding_size).unsqueeze(0)
        return None

    def _gaussian3D(self, radius, sigma=2, dtype=torch.float32, device="cpu"):
        x = torch.arange(-radius, radius + 1, dtype=dtype, device=device).view(-1, 1, 1)
        y = torch.arange(-radius, radius + 1, dtype=dtype, device=device).view(1, -1, 1)
        z = torch.arange(-radius, radius + 1, dtype=dtype, device=device).view(1, 1, -1)

        h = (-(x * x + y * y + z * z) / (2 * sigma * sigma)).exp()

        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        return h

    def _points_to_clickmap(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Points to click map"""
        bs, num_pts = points.shape[0], points.shape[1]
        pos_map = torch.zeros((bs, ) + (1, ) + self.input_image_size, device=points.device)
        neg_map = torch.zeros((bs, ) + (1, ) + self.input_image_size, device=points.device)
        radius = 8
        gaussian_kernel = self._gaussian3D(radius, sigma=2, device=points.device)

        for i in range(bs):
            for j in range(num_pts):
                # check valid point
                if labels[i, j] == -1:
                    continue
                center_v = (points[i, j, 0], points[i, j, 1], points[i, j, 2])
                vx, vy, vz = center_v
                vx = vx.long()
                vy = vy.long()
                vz = vz.long()
                depth, height, width = (self.input_image_size[2], self.input_image_size[1], self.input_image_size[0])

                right, left = min(vx, radius), min(width - vx, radius + 1)
                anterior, posterior = min(vy, radius), min(height - vy, radius + 1)
                inferior, superior = min(vz, radius), min(depth - vz, radius + 1)
                if labels[i, j] == 1:
                    pos_map[i, :,
                    vz - inferior : vz + superior,
                    vy - anterior : vy + posterior,
                    vx - right : vx + left,
                    ] = gaussian_kernel[
                    radius - inferior : radius + superior,
                    radius - anterior : radius + posterior,
                    radius - right : radius + left,
                    ]
                else:
                    neg_map[i, :,
                    vz - inferior : vz + superior,
                    vy - anterior : vy + posterior,
                    vx - right : vx + left,
                    ] = gaussian_kernel[
                    radius - inferior : radius + superior,
                    radius - anterior : radius + posterior,
                    radius - right : radius + left,
                    ]

        # normalization
        pos_map = self._normalization(pos_map)
        neg_map = self._normalization(neg_map)

        click_map = torch.concat([pos_map, neg_map], dim=1)
        return click_map

    def _normalization(self, click_map):
        if torch.sum(click_map) == 0:
            return click_map
        else:
            click_map_max = torch.max(click_map)
            click_map_min = torch.min(click_map)

            norm_click_map = (click_map - click_map_min) / (click_map_max - click_map_min)

            return norm_click_map
    
    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_D)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        device = points[0].device if points is not None else masks.device
        click_map = torch.zeros((bs, ) + (2, ) + self.input_image_size, device=device)
        if points is not None:
            coords, labels = points
            click_map = self._points_to_clickmap(coords, labels)

        if masks is not None:
            dense_embeddings = masks
        else:
            dense_embeddings = torch.zeros((bs, 1, self.input_image_size[0], self.input_image_size[1], self.input_image_size[2]), device=coords.device)

        return click_map, dense_embeddings
    

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        d, h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((d, h, w), device=device, dtype=torch.float32)
        z_embed = grid.cumsum(dim=0) - 0.5
        y_embed = grid.cumsum(dim=1) - 0.5
        x_embed = grid.cumsum(dim=2) - 0.5
        z_embed = z_embed / d
        y_embed = y_embed / h
        x_embed = x_embed / w
        
        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x D x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1].
        coords_input: [X, Y, Z] format.
        image_size: [D, H, W] format.
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[2]
        coords[:, :, 1] = coords[:, :, 1] / image_size[1]
        coords[:, :, 2] = coords[:, :, 2] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
