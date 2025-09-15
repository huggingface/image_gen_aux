from abc import abstractmethod
from typing import Union
import numpy as np
import PIL.Image
import torch

from ..base_model_processor import BaseModelProcessor


class Interpolater(BaseModelProcessor):
    """
    Abstract base class for all interpolation models.
    Defines a common API contract.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Interpolates between images or frames.

        Child classes must implement this.
        """
        pass