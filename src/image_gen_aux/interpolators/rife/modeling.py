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

import functools
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_mixin import ModelMixin, register_to_config


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        out_channels: int = 8,
        nonlinearity: Type[nn.Module] = nn.LeakyReLU,
    ) -> None:
        super().__init__()

        blocks = []
        blocks.append(nn.Conv2d(in_channels, hidden_channels, 3, 2, 1))
        blocks.append(nonlinearity())
        blocks.append(nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1))
        blocks.append(nonlinearity())
        blocks.append(nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1))
        blocks.append(nonlinearity())
        blocks.append(nn.ConvTranspose2d(hidden_channels, out_channels, 4, 2, 1))
        blocks.append(nonlinearity())
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        nonlinearity: Type[nn.Module] = nn.LeakyReLU,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.beta = nn.Parameter(torch.ones((1, out_channels, 1, 1)), requires_grad=True)
        self.nonlinearity = nonlinearity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states * self.beta
        hidden_states = hidden_states + residual
        hidden_states = self.nonlinearity(hidden_states)

        return hidden_states


class IntermediateFlowBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: Optional[int] = None,
        num_hidden_blocks: int = 8,
        nonlinearity: Type[nn.Module] = nn.LeakyReLU,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 3, 2, 1),
            nonlinearity(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 3, 2, 1),
            nonlinearity(),
        )

        blocks = []
        for _ in range(num_hidden_blocks):
            blocks.append(ResnetBlock(hidden_channels, hidden_channels, 3, 1, 1, nonlinearity))
        self.blocks = nn.ModuleList(blocks)

        self.conv_output = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(
        self, hidden_states: torch.Tensor, flow: Optional[torch.Tensor] = None, scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = F.interpolate(hidden_states, scale_factor=1 / scale, mode="bilinear", align_corners=False)

        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1 / scale, mode="bilinear", align_corners=False) / scale
            hidden_states = torch.cat([hidden_states, flow], dim=1)

        hidden_states = self.conv_input(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.conv_output(hidden_states)

        hidden_states = F.interpolate(hidden_states, scale_factor=scale, mode="bilinear", align_corners=False)
        flow, mask, features = hidden_states.split_with_sizes([4, 1, hidden_states.size(1) - 5], dim=1)
        flow = flow * scale

        return flow, mask, features


class IntermediateFlowNet(ModelMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: List[int] = [7 + 16, 8 + 4 + 16 + 8, 8 + 4 + 16 + 8, 8 + 4 + 16 + 8],
        hidden_channels: List[int] = [256, 192, 96, 48],
        out_channels: List[int] = [52, 52, 52, 52],
        encoder_in_channels: int = 3,
        encoder_out_channels: int = 32,
        num_hidden_blocks: int = 8,
        nonlinearity: Type[nn.Module] = functools.partial(nn.LeakyReLU, negative_slope=0.2),
    ) -> None:
        super().__init__()

        self.encoder = Encoder(encoder_in_channels, encoder_out_channels, 8, nonlinearity)

        blocks = []
        for in_channel, hidden_channel, out_channel in zip(in_channels, hidden_channels, out_channels):
            blocks.append(
                IntermediateFlowBlock(in_channel, hidden_channel, out_channel, num_hidden_blocks, nonlinearity)
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, image_1: torch.Tensor, image_2: torch.Tensor, timestep: float = 0.5, scale: List[float] = [8, 4, 2, 1]
    ) -> torch.Tensor:
        batch_size, channels, height, width = image_1.shape
        assert image_1.shape == image_2.shape

        timestep = image_1.new_full((1, 1, height, width), timestep)
        warped_image_1 = image_1
        warped_image_2 = image_2

        hidden_states_1 = self.encoder(image_1)
        hidden_states_2 = self.encoder(image_2)

        flow, mask, features = None, None, None

        for i, block in enumerate(self.blocks):
            if flow is None:
                hidden_states = torch.cat([image_1, image_2, hidden_states_1, hidden_states_2, timestep], dim=1)
                flow, mask, features = block(hidden_states, None, scale[i])
            else:
                flow_1, flow_2 = flow.split(2, dim=1)
                warped_flow_1 = _warp(hidden_states_1, flow_1)
                warped_flow_2 = _warp(hidden_states_2, flow_2)
                hidden_states = torch.cat(
                    [warped_image_1, warped_image_2, warped_flow_1, warped_flow_2, timestep, mask, features], dim=1
                )
                flow_, mask, features = block(hidden_states, flow, scale[i])
                flow = flow + flow_

            flow_1, flow_2 = flow.split(2, dim=1)

        warped_image_1 = _warp(image_1, flow_1)
        warped_image_2 = _warp(image_2, flow_2)

        mask = mask.sigmoid()
        output = warped_image_1 * mask + warped_image_2 * (1 - mask)
        return output


def _warp(hidden_states: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = flow.shape

    flow_1 = flow[:, 0:1, :, :] / ((width - 1) / 2)
    flow_2 = flow[:, 1:2, :, :] / ((height - 1) / 2)
    flow = torch.cat([flow_1, flow_2], dim=1)

    horizontal = torch.linspace(-1, 1, width, device=flow.device, dtype=flow.dtype).view(1, 1, 1, width)
    horizontal = horizontal.expand(batch_size, -1, height, -1)
    vertical = torch.linspace(-1, 1, height, device=flow.device, dtype=flow.dtype).view(1, 1, height, 1)
    vertical = vertical.expand(batch_size, -1, -1, width)
    grid = torch.cat([horizontal, vertical], dim=1)
    grid = grid + flow
    grid = grid.permute(0, 2, 3, 1)

    warped_flow = F.grid_sample(hidden_states, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return warped_flow
