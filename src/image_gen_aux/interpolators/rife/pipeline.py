# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL.Image import Image

from ...image_processor import ImageMixin
from ...utils.loss import ssim
from .modeling import IntermediateFlowNet


class RIFEPipeline(ImageMixin):
    def __init__(self, flownet: IntermediateFlowNet) -> None:
        super().__init__()

        # TODO: register models
        self.flownet = flownet

    def check_inputs(self, video) -> None:
        if not isinstance(video, list):
            raise ValueError("`video` must be a list of PIL images.")

        num_frames = len(video)
        if num_frames < 2:
            raise ValueError("`video` must have atleast 2 frames.")
        if not isinstance(video[0], Image):
            raise ValueError("`video` must be a list of PIL images.")

    @torch.no_grad()
    def __call__(
        self,
        video: List[Image],
        scale: float = 1.0,
        target_fps_multiple: int = 2,
        downscaled_size: Union[int, Tuple[int, int]] = (32, 32),
        ssim_repeat_threshold: float = 0.2,
        output_type: Union[List[Image], np.ndarray, torch.Tensor] = "pil",
    ) -> Union[List[Image], np.ndarray, torch.Tensor]:
        self.check_inputs(video)

        num_frames = len(video)
        width = video[0].width
        height = video[0].height
        dtype = self.flownet.dtype
        device = self.flownet.device

        tmp = max(128, int(128 / scale))
        padding_height = ((height - 1) // tmp + 1) * tmp
        padding_width = ((width - 1) // tmp + 1) * tmp
        padding = (0, padding_width - width, 0, padding_height - height)

        video = self.convert_image_to_tensor(video)
        video = video.to(device=device, dtype=dtype)
        video = F.pad(video, padding, mode="constant", value=0)
        video_downscaled = F.interpolate(video, size=downscaled_size, mode="bilinear", align_corners=False)

        scale = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        output_video = []

        for i in range(num_frames - 1):
            frame_0 = video[i].unsqueeze(0)
            frame_1 = video[i + 1].unsqueeze(0)
            frame_0_downscaled = video_downscaled[i].unsqueeze(0)
            frame_1_downscaled = video_downscaled[i + 1].unsqueeze(0)
            ssim_loss = ssim(frame_0_downscaled, frame_1_downscaled)

            if ssim_loss < ssim_repeat_threshold:
                interpolated_frames = [frame_0] * (target_fps_multiple - 1)
            else:
                interpolated_frames = []
                for j in range(target_fps_multiple - 1):
                    interpolated_frames.append(
                        self.flownet(frame_0, frame_1, (j + 1) / (target_fps_multiple + 1), scale=scale)
                    )

            output_video.append(frame_0)
            output_video.extend(interpolated_frames)

        output_video.append(video[-1].unsqueeze(0))
        output_video = torch.cat(output_video, dim=0)[:, :, :height, :width]
        output_video = self.post_process_image(output_video, output_type)

        return output_video
