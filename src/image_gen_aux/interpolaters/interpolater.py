from abc import abstractmethod
from typing import Union, List
import numpy as np
import PIL.Image
import torch

from ..base_model_processor import BaseModelProcessor


class FrameInterpolater(BaseModelProcessor):
    """
    Abstract base class for all interpolation models.
    Defines a common API contract.
    """
    
    def __init__(self):
        """
        Private constructor. Use from_images() or from_video() class methods instead.
        """
        super().__init__()
        self._input_type = None
        self._images = None
        self._video_path = None
    
    @classmethod
    def from_images(cls, image1: Union[str, PIL.Image.Image, np.ndarray, torch.Tensor], 
                   image2: Union[str, PIL.Image.Image, np.ndarray, torch.Tensor]):
        """
        Create interpolater from two images.
        
        Args:
            image1: First image (path, PIL Image, numpy array, or torch tensor)
            image2: Second image (path, PIL Image, numpy array, or torch tensor)
        
        Returns:
            Interpolater instance configured for image interpolation
        """
        instance = cls()
        instance._input_type = "images"
        instance._images = (image1, image2)
        return instance
    
    @classmethod
    def from_video(cls, video_path: str):
        """
        Create interpolater from video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Interpolater instance configured for video interpolation
        """
        instance = cls()
        instance._input_type = "video"
        instance._video_path = video_path
        return instance
    
    @property
    def input_type(self) -> str:
        """Get the input type ('images' or 'video')."""
        return self._input_type
    
    @property
    def images(self) -> tuple:
        """Get the input images (only available for image input type)."""
        if self._input_type != "images":
            raise ValueError("Images only available when input type is 'images'")
        return self._images
    
    @property
    def video_path(self) -> str:
        """Get the video path (only available for video input type)."""
        if self._input_type != "video":
            raise ValueError("Video path only available when input type is 'video'")
        return self._video_path

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Interpolates between images or frames.
        
        Child classes must implement this method and handle both input types:
        - When input_type == "images": interpolate between the two images
        - When input_type == "video": interpolate frames in the video
        """
        pass
