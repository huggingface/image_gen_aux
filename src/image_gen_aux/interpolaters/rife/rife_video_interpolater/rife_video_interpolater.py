import cv2
from typing import List, Union
import numpy as np
import PIL.Image
import torch

from .interpolater import Interpolater
from .rife_image_interpolater import RIFEImageInterpolator


class RIFEVideoInterpolator(Interpolater):
    """
    Interpolates frames for a full video using RIFE.
    Wraps the RIFEImageInterpolator in a loop.
    """

    def __init__(self, model):
        super().__init__()
        self.image_interpolator = RIFEImageInterpolator(model)

    def __call__(
        self,
        frames: List[Union[PIL.Image.Image, np.ndarray, torch.Tensor]],
        return_type: str = "pil",
    ):
        """
        Args:
            frames: list of consecutive video frames
            return_type: output type ("pil", "np", "pt")

        Returns:
            list of interpolated frames (same type as return_type)
        """
        outputs = []
        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            mid = self.image_interpolator(f1, f2, return_type=return_type)
            outputs.append(f1)
            outputs.append(mid)
        outputs.append(frames[-1])
        return outputs
