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

from typing import Optional

import torch
import torch.nn.functional as F


def _gaussian(size: int, sigma: float) -> torch.Tensor:
    g = [-((i - size // 2) ** 2) / float(2 * sigma**2) for i in range(size)]
    g = torch.tensor(g)
    g = torch.exp(g)
    g = g / g.sum()
    return g


def _create_window_2d(size: int, channels: int = 1) -> torch.Tensor:
    window_1d = _gaussian(size, 1.5).unsqueeze(1)
    window_2d = torch.matmul(window_1d, window_1d.t())
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous()
    return window


def _create_window_3d(size: int, channels: int = 1) -> torch.Tensor:
    window_1d = _gaussian(size, 1.5).unsqueeze(1)
    window_2d = torch.matmul(window_1d, window_1d.t()).unsqueeze(2)
    window_3d = torch.matmul(window_2d, window_1d.t())
    window = window_3d.expand(1, channels, size, size, size).contiguous()
    return window


def ssim(
    image_1: torch.Tensor, image_2: torch.Tensor, window_size: int = 11, window: Optional[torch.Tensor] = None
) -> torch.Tensor:
    max_val = 1
    min_val = 0
    L = max_val - min_val

    padding = 0
    batch_size, channels, height, width = image_1.shape
    if window is None:
        real_size = min(window_size, height, width)
        window = _create_window_3d(real_size, channels=1).to(image_1.device)

    # Channel is set to 1 since we consider color images as volumetric images
    image_1 = image_1.unsqueeze(1)
    image_2 = image_2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(image_1, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padding)
    mu2 = F.conv3d(F.pad(image_2, (5, 5, 5, 5, 5, 5), mode="replicate"), window, padding=padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(image_1 * image_1, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padding) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(image_2 * image_2, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padding) - mu2_sq
    sigma = F.conv3d(F.pad(image_1 * image_2, (5, 5, 5, 5, 5, 5), "replicate"), window, padding=padding) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma + C2
    v2 = sigma1_sq + sigma2_sq + C2
    # contrast_sensitivity = torch.mean(v1 / v2)

    ssim = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ssim = ssim.mean()
    return ssim
