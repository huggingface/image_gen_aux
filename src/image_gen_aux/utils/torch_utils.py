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

import torch


_DTYPE_MAPPING = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "torch.float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
}


def get_torch_dtype_from_string(dtype: str) -> torch.dtype:
    if dtype not in _DTYPE_MAPPING.keys():
        raise ValueError(f"The data type {dtype=} is invalid. Supported dtypes: {list(_DTYPE_MAPPING.keys())}")
    return _DTYPE_MAPPING[dtype]
