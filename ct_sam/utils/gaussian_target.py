# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn.functional as F


def gaussian3D(radius, sigma=2, dtype=torch.float32, device="cpu"):
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device).view(-1, 1, 1)
    y = torch.arange(-radius, radius + 1, dtype=dtype, device=device).view(1, -1, 1)
    z = torch.arange(-radius, radius + 1, dtype=dtype, device=device).view(1, 1, -1)

    h = (-(x * x + y * y + z * z) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernel.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool3d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores, k=24):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, _, width, height, depth = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (width * height * depth)
    topk_inds = topk_inds % (width * height * depth)

    topk_xs = topk_inds // (height * depth)
    topk_ys = (topk_inds % (height * depth)) // depth
    topk_zs = (topk_inds % (height * depth)) % depth
    return topk_scores, topk_inds, topk_clses, topk_xs, topk_ys, topk_zs


def gather_feat(feat, ind):
    """Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.size()[-1]
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    return feat


def transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 4, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(4))
    feat = gather_feat(feat, ind)
    return feat
