import numpy as np
import SimpleITK as sitk
from typing import Optional


def normalize_sitk_im(
    sitk_im: sitk.Image,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    clip: bool = True,
    min_out_is_zero: bool = False,
) -> sitk.Image:
    """
    normalize a simpleitk image, if either min_value or max_value is not given, will use the
    intensities at 1 and 99 percentiles to adaptively normalize the image

    :param sitk_im: the input image to be normalized
    :param min_value: min value to use
    :param max_value: max value to use
    :param clip: whether clip values out of the range, defaults to True
    :return: the normalized image
    """
    pixel_type = sitk_im.GetPixelID()
    if pixel_type not in [sitk.sitkFloat32, sitk.sitkFloat64]:
        raise TypeError("the dtype of the image to be normalized should be float32 or float64!")

    im_np = sitk.GetArrayFromImage(sitk_im)
    if min_value is None:
        min_value = np.percentile(im_np, 1)
    if max_value is None:
        max_value = np.percentile(im_np, 99)
    if clip:
        sitk_im = sitk.Clamp(sitk_im, lowerBound=min_value, upperBound=max_value)

    shift = -min_value
    scale = 1 / (max_value - min_value)
    if min_out_is_zero:
        sitk_im = sitk.ShiftScale(sitk_im, shift, scale)
    else:
        shift -= (max_value - min_value) / 2
        scale *= 2
        sitk_im = sitk.ShiftScale(sitk_im, shift, scale)

    return sitk_im
