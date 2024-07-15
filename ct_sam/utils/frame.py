# Copyright (c) DAMO Health

import numpy as np
import SimpleITK as sitk
import torch
from typing import Dict, List, Tuple, Union


class Frame3d(object):
    """
    frame class for 3d image, with infomation of origin, spacing, direction

    """

    def __init__(self, ref_image: sitk.Image = None):
        super(Frame3d, self).__init__()
        if ref_image is None:
            self.from_dict({})
        else:
            assert isinstance(
                ref_image, sitk.Image
            ), "ref_image must be None or a SimpleITK Image instance!"
            self._origin = ref_image.GetOrigin()
            self._spacing = ref_image.GetSpacing()
            self._direction = ref_image.GetDirection()

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, value: Union[List[float], Tuple[float]]):
        value = np.array(value).reshape(-1).tolist()
        if len(value) != 3:
            raise ValueError("origin must be a 3d vector")
        self._origin = value

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, value: Union[List[float], Tuple[float]]):
        value = np.array(value).reshape(-1).tolist()
        if len(value) != 3:
            raise ValueError("spacing must be a 3d vector")
        self._spacing = value

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value: Union[List[float], Tuple[float]]):
        value = np.array(value).reshape(-1).tolist()
        if len(value) != 9:
            raise ValueError("direction must be a 9d vector")
        self._direction = value

    def voxel_to_world(self, voxel_coord: Union[List, Tuple, np.ndarray, torch.Tensor]):
        """
        transforms the voxel_coord to world_coord according to current frame information
        origin : position of the first voxel in the world coordinates(vector denoted Origin)
        spacing : size of the voxels in mm (vector denoted S)
        direction () : direction cosine (square matrix denoted D)
        (x,y,z)^T = D * S * (i,j,k)^T + Origin

        :param voxel_coord: a list of three elements or a nd-array of shape [3, N],
         the voxel coordinate need to be transformed
        :return: the output world_coord
        """
        if len(voxel_coord) != 3:
            raise ValueError("voxel_coord must be a 3d vector or an array of shape [3, N]")

        D = np.array(self._direction, dtype=np.float32).reshape([3, 3])
        S = np.array(self._spacing, dtype=np.float32).reshape([3, 1])
        Origin = np.array(self._origin, dtype=np.float32).reshape([3, 1])
        S = np.eye(3, dtype=np.float32) * S
        DS = D @ S

        if isinstance(voxel_coord, torch.Tensor):
            DS = torch.from_numpy(DS).to(voxel_coord.device)
            Origin = torch.from_numpy(Origin).to(voxel_coord.device)
            voxel_coord = voxel_coord.type(torch.float32)
        else:
            voxel_coord = np.array(voxel_coord, np.float32).reshape([3, -1])
        world_coord = DS @ voxel_coord + Origin
        if world_coord.shape[1] == 1:
            world_coord = world_coord.reshape(-1)
        return world_coord

    def world_to_voxel(
        self, world_coord: Union[List[float], Tuple[float], np.ndarray, torch.Tensor], is_round=True
    ):
        """
        transforms the world_coord to voxel_coord according to current frame information
        (i,j,k)^T = ( D * S )^(-1) * ((x,y,z)^T - Origin)

        :param world_coord: a list of three elements or a nd-array of shape [3, N],
        the world_coord need to be transformed
        :return: a list, the output voxel_coord
        """
        if len(world_coord) != 3:
            raise ValueError("world_coord must be a 3d vector or an array of shape [3, N]")

        D = np.array(self._direction, dtype=np.float32).reshape([3, 3])
        S = np.array(self._spacing, dtype=np.float32).reshape([3, 1])
        Origin = np.array(self._origin, dtype=np.float32).reshape([3, 1])
        S = np.eye(3, dtype=np.float32) * S
        INVDS = np.linalg.inv(D @ S)

        if isinstance(world_coord, torch.Tensor):
            INVDS = torch.from_numpy(INVDS).to(world_coord.device)
            Origin = torch.from_numpy(Origin).to(world_coord.device)
        else:
            world_coord = np.array(world_coord).reshape([3, -1])
        voxel_coord = INVDS @ (world_coord - Origin)
        if voxel_coord.shape[1] == 1:
            voxel_coord = voxel_coord.reshape(-1)
        if is_round and not isinstance(voxel_coord, torch.Tensor):
            voxel_coord = voxel_coord.round().astype(np.int32)
        return voxel_coord

    def from_dict(self, frame_dict: Dict):
        self.origin = frame_dict.get("origin", (0.0, 0.0, 0.0))
        self.spacing = frame_dict.get("spacing", (1.0, 1.0, 1.0))
        self.direction = frame_dict.get("direction", (1, 0, 0, 0, 1, 0, 0, 0, 1))

    def to_dict(self) -> Dict:
        return dict(origin=self.origin, spacing=self.spacing, direction=self.direction)

    def to_numpy(self) -> Dict:
        """
        in numpy + zyx order
        """
        # lack of better name
        return dict(
            origin=np.array(self.origin[::-1]),
            spacing=np.array(self.spacing[::-1]),
            direction=np.array(self.direction),
        )

    def __repr__(self):
        return str(self.__class__) + ": " + str(self.to_dict())

    def clone(self):
        new_frame = Frame3d()
        new_frame.from_dict(self.to_dict())
        return new_frame


def set_frame_as_ref(im: sitk.Image, ref_im: sitk.Image):
    """
    copy frame infomation from reference image

    :param im: destination image
    :param ref_im: reference image
    :return: None
    """
    im.SetOrigin(ref_im.GetOrigin())
    im.SetSpacing(ref_im.GetSpacing())
    im.SetDirection(ref_im.GetDirection())


def set_frame(im: sitk.Image, frame: Frame3d):
    """
    set the frame of an image to a new one

    :param im: destination image
    :param frame: the new frame to use
    :return: None
    """
    im.SetOrigin(frame.origin)
    im.SetSpacing(frame.spacing)
    im.SetDirection(frame.direction)


def voxel_to_world(sitk_im, voxel_coord: Union[List, Tuple, np.ndarray]):
    """
    this function transforms the voxel_coord to world_coord according to an image's frame

    :param sitk_im: the reference image
    :param voxel_coord: a list or a ndarray of three elements, the voxel_coord need
           to be transformed
    :return: a list, the output world_coord
    """
    frame = Frame3d(sitk_im)
    return frame.voxel_to_world(voxel_coord)


def world_to_voxel(
    sitk_im, world_coord: Union[List[float], Tuple[float], np.ndarray], is_round=True
):
    """
    this function transforms the world_coord to voxel_coord according to an image's frame

    :param sitk_im: the reference image
    :param world_coord: a list or a nd-array of three elements, the world_coord need
           to be transformed
    :param is_round: return rounded int voxel coord if True
    :return: a list, the output voxel_coord
    """
    frame = Frame3d(sitk_im)
    return frame.world_to_voxel(world_coord, is_round)


def world_box(sitk_im: sitk.Image):
    """
    get the bounding box of sitk image in world coordinate system

    :param sitk_im: the input sitk image
    :return: two lists, each contains three elements, denoting the min_coord and max_coord of
             the box in world_coord, respectively.
    """
    start = [0, 0, 0]
    end = sitk_im.GetSize()
    coords = list(zip(start, end))
    corners = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                voxel_coord = [coords[0][i], coords[1][j], coords[2][k]]
                corners.append(voxel_to_world(sitk_im, voxel_coord))
    corners = np.vstack(corners)
    min_corner = corners.min(axis=0)
    max_corner = corners.max(axis=0)
    return min_corner, max_corner


def get_frame_from_file(im_path: str):
    reader = sitk.ImageFileReader()
    reader.SetFileName(im_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    frame = Frame3d()
    frame.origin = reader.GetOrigin()
    frame.spacing = reader.GetSpacing()
    frame.direction = reader.GetDirection()

    return frame
