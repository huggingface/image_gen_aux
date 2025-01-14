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

import argparse
from typing import Any, Callable, Dict, Optional

import torch.nn as nn


def drop_(state_dict: Dict[str, Any], key: str) -> None:
    state_dict.pop(key, None)


def remap_encode_(state_dict: Dict[str, Any], key: str) -> None:
    # encode.cnn{x}.(weight|bias) -> encoder.blocks.{2*x}.(weight|bias)
    split_key = key.split(".")
    new_key = f"encoder.blocks.{2 * int(split_key[1][-1])}.{'.'.join(split_key[2:])}"
    state_dict[new_key] = state_dict.pop(key)


_REPLACE_KEYS_DICT = {
    "module.": "",
    "convblock": "blocks",
    "block0": "blocks.0",
    "block1": "blocks.1",
    "block2": "blocks.2",
    "block3": "blocks.3",
    "lastconv": "conv_output",
    "conv0.0.0": "conv_input.0",
    "conv0.1.0": "conv_input.2",
}

_CUSTOM_KEYS_REMAP_DICT = {
    "teacher": drop_,
    "caltime": drop_,
    "encode": remap_encode_,
}

_COMMON_MODELING_KEY_NAMES = ["model", "module", "state_dict", "pytorch_model", "pytorch_module"]


def recursively_find_state_dict(possibly_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    for name in _COMMON_MODELING_KEY_NAMES:
        if name in possibly_state_dict.keys():
            return recursively_find_state_dict(possibly_state_dict[name])
    return possibly_state_dict


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def replace_keys_(state_dict: Dict[str, Any], replace_keys: Dict[str, str]) -> None:
    for key in list(state_dict.keys()):
        renamed_key = key
        for old_key, new_key in replace_keys.items():
            renamed_key = renamed_key.replace(old_key, new_key)
        update_state_dict_(state_dict, key, renamed_key)


def custom_remap_(
    state_dict: Dict[str, Any], custom_remap_keys: Dict[str, Callable[[Dict[str, Any], str], None]]
) -> None:
    for identifier, remap_fn_ in list(custom_remap_keys.items()):
        for key in list(state_dict.keys()):
            if identifier not in key:
                continue
            remap_fn_(state_dict, key)


def convert(
    original_ckpt_path: str, output_dir: Optional[str] = None, dtype: str = "fp32", push_to_hub: bool = False
) -> nn.Module:
    from image_gen_aux.interpolators import IntermediateFlowNet
    from image_gen_aux.interpolators.utils.model_loading_utils import load_state_dict
    from image_gen_aux.utils.torch_utils import get_torch_dtype_from_string

    torch_load_kwargs = {"map_location": "cpu", "weights_only": True}
    state_dict = load_state_dict(original_ckpt_path, torch_load_kwargs)
    state_dict = recursively_find_state_dict(state_dict)

    replace_keys_(state_dict, _REPLACE_KEYS_DICT)
    custom_remap_(state_dict, _CUSTOM_KEYS_REMAP_DICT)

    model = IntermediateFlowNet()
    model.load_state_dict(state_dict, strict=True)

    dtype = get_torch_dtype_from_string(dtype)
    model.to(dtype=dtype)

    if output_dir is not None:
        model.save_pretrained(output_dir)

    if push_to_hub:
        # TODO
        model.push_to_hub()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_ckpt_path", type=str, required=True, help="Path to original checkpoint.")
    parser.add_argument(
        "--output_dir", type=str, help="Path to output directory where converted model should be stored."
    )
    parser.add_argument("--dtype", type=str, default="fp32", help="Data type in which the model should be stored.")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted checkpoint to the Hub."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    convert(
        original_ckpt_path=args.original_ckpt_path,
        output_dir=args.output_dir,
        dtype=args.dtype,
        push_to_hub=args.push_to_hub,
    )
