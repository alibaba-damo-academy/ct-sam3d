# Copyright (c) DAMO Health

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from typing import Iterable, List, Tuple, Union

from ct_sam.utils.frame import Frame3d, world_box


def resample_itkimage_torai(
    sitk_im: sitk.Image,
    spacing: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    resample an image to RAI coordinate system

    :param sitk_im: input image
    :param spacing: destination spacing
    :param interpolator: interpolation method, can be 'nearest' and linear, defaults to 'nearest'
    :param pad_value: value for voxels extroplated
    :return: the resampled SimpleITK image object
    """

    min_corner, max_corner = world_box(sitk_im)

    origin = min_corner.tolist()
    direction = np.eye(3, dtype=np.double).flatten().tolist()
    size = ((max_corner - min_corner) / (np.array(spacing))).round().astype(np.int32).tolist()

    return resample_base(
        sitk_im,
        origin=origin,
        direction=direction,
        spacing=spacing,
        size=size,
        interpolator=interpolator,
        pad_value=pad_value,
    )


def flip_itkimage_torai(
    sitk_im: sitk.Image, interpolator: str = "nearest", pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    flip axes or transpose to make non-RAI image to RAI, with original spacings in each axis
    """
    or_code = get_itk_img_orientation_code(sitk_im)
    if "Oblique" in or_code:
        or_code = or_code[-4:-1]
    or_code = or_code.replace("L", "R").replace("P", "A").replace("S", "I")
    order = []
    for s in "RAI":
        order.append(or_code.index(s))
    spacing = np.array(sitk_im.GetSpacing())
    spacing = spacing[order]
    return resample_itkimage_torai(sitk_im, spacing, interpolator, pad_value)


def resample_itkimage_withsize(
    itkimage: sitk.Image,
    new_size: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    Image resize with size by sitk resampleImageFilter.

    :param itkimage: input itk image or itk volume.
    :param new_size: the target size of the resampled image, such as [120, 80, 80].
    :param interpolator: for mask used nearest, for image linear is an option.
    :param pad_value: the value for the pixel which is out of image.
    :return: resampled itk image.
    """

    # get resize factor
    origin_size = np.array(itkimage.GetSize())
    new_size = np.array(new_size)
    factor = origin_size / new_size

    # get new spacing
    origin_spcaing = itkimage.GetSpacing()
    new_spacing = factor * origin_spcaing

    itkimg_resampled = resample_base(
        itkimage,
        itkimage.GetOrigin(),
        itkimage.GetDirection(),
        new_spacing,
        new_size,
        interpolator,
        pad_value,
    )

    return itkimg_resampled


def resample_itkimage_withspacing(
    itkimage: sitk.Image,
    new_spacing: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    Image resize with size by sitk resampleImageFilter.

    :param itkimage: input itk image or itk volume.
    :param new_spacing: the target spacing of the resampled image, such as [1.0, 1.0, 1.0].
    :param interpolator: for mask used nearest, for image linear is an option.
    :param pad_value: the value for the pixel which is out of image.
    :return: resampled itk image.
    """

    # get resize factor
    origin_spacing = itkimage.GetSpacing()
    new_spacing = np.array(new_spacing, float)
    factor = new_spacing / origin_spacing

    # get new image size
    origin_size = itkimage.GetSize()
    new_size = origin_size / factor
    new_size = new_size.astype(np.int)

    itkimg_resampled = resample_base(
        itkimage,
        itkimage.GetOrigin(),
        itkimage.GetDirection(),
        new_spacing,
        new_size,
        interpolator,
        pad_value,
    )

    return itkimg_resampled


def resample_base(
    sitk_im: sitk.Image,
    origin: Union[List, Tuple, np.ndarray],
    direction: Union[List, Tuple],
    spacing: Union[List, Tuple, np.ndarray],
    size: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    the base resample function, can be used to resample a small patch out of the original image
    or to resample to sample patch back to the original image, and of course, to resize a volume

    :param sitk_im: input image
    :param origin: the origin of the resampled volume
    :param direction: the direction of the resampled volume
    :param spacing: the spacing of the resampled volume
    :param size: the output size of the resampled volume
    :param interpolator: interpolation method, can be 'nearest' and linear, defaults to 'nearest'
    :param pad_value: value for voxels extroplated
    :return: the resampled SimpleITK image object
    """
    size = [int(s) for s in size]
    SITK_INTERPOLATOR_DICT = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "gaussian": sitk.sitkGaussian,
        "label_gaussian": sitk.sitkLabelGaussian,
        "bspline": sitk.sitkBSpline,
        "hamming_sinc": sitk.sitkHammingWindowedSinc,
        "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
        "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
        "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
    }

    assert (
        interpolator in SITK_INTERPOLATOR_DICT.keys()
    ), "`interpolator` should be one of {}".format(SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = SITK_INTERPOLATOR_DICT[interpolator]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(size)
    resample_filter.SetOutputSpacing(np.array(spacing).tolist())
    resample_filter.SetOutputOrigin(np.array(origin).tolist())
    resample_filter.SetOutputDirection(direction)
    resample_filter.SetOutputPixelType(sitk_im.GetPixelID())
    resample_filter.SetDefaultPixelValue(pad_value)
    resample_filter.SetInterpolator(sitk_interpolator)

    img = resample_filter.Execute(sitk_im)

    return img


def resample_itkimage_withspacing_by_torch(
    itkimage: sitk.Image,
    new_spacing: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:

    mode = "nearest"
    if interpolator == "linear":
        mode = "trilinear"

    dtype_conversion = False
    if itkimage.GetPixelID() != 8:
        img = sitk.Cast(itkimage, sitk.sitkFloat32)
        dtype_conversion = True
    else:
        img = itkimage

    im_tensor = torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available():
        im_tensor = im_tensor.cuda()
    ratio = np.array(itkimage.GetSpacing()) / new_spacing
    ratio = ratio[::-1]
    im_tensor = F.interpolate(im_tensor, scale_factor=ratio, mode=mode, align_corners=False)
    im_tensor = im_tensor.squeeze().cpu()
    img = sitk.GetImageFromArray(im_tensor.numpy())
    img.SetSpacing(np.array(new_spacing).tolist())
    img.SetOrigin(itkimage.GetOrigin())

    if dtype_conversion:
        img = sitk.Cast(img, itkimage.GetPixelID())

    return img


def crop_roi_with_center(
    itk_img: sitk.Image,
    center_w: Iterable[float],
    spacing: Iterable[float],
    x_axis: Iterable[float],
    y_axis: Iterable[float],
    z_axis: Iterable[float],
    size: Iterable[int],
    interpolator: str = "nearest",
    pad_value=-1024,
):
    frame = Frame3d()
    frame.origin = list(center_w)
    frame.direction = np.vstack([x_axis, y_axis, z_axis]).transpose().flatten().tolist()
    frame.spacing = spacing
    size = np.array(size).reshape(3)
    true_origin = frame.voxel_to_world(-size / 2)
    roi = resample_base(
        itk_img, true_origin, frame.direction, frame.spacing, size, interpolator, pad_value
    )
    return roi
