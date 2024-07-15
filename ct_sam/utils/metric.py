import torch
import torch.nn.functional as F


def compute_dsc(predictions, targets, threshold=0.0, smooth=1):
    """
    compute Dice coefficient

    :param predictions: Nd tensor of predicted logits, need sigmoid.
    :param targets: Nd tensor of ground-truth mask, D * H * W.
    :param threshold: binarization threshold.
    :return: DSC, FP positions, and FN positions
    """
    predictions = F.sigmoid(predictions).detach()
    predict = predictions > threshold
    
    intersection = (predict * targets).sum()
    
    dice = (2.0 * intersection + smooth) / (predict.sum() + targets.sum() + smooth)
    fp = torch.nonzero((predict == 1) * (targets == 0), as_tuple=True)
    fn = torch.nonzero((predict == 0) * (targets == 1), as_tuple=True)
    
    return dice, fp, fn
    