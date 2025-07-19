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

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Union

import safetensors
import torch
from huggingface_hub.utils import EntryNotFoundError

from ...utils.constants import SAFETENSORS_FILE_EXTENSION, SAFETENSORS_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME
from .hub_utils import _add_variant, _get_model_file


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike], torch_load_kwargs: Dict[str, Any] = {}
) -> Dict[str, Any]:
    r"""Reads a checkpoint file, returning properly formatted errors if they arise."""

    try:
        file_extension = os.path.basename(checkpoint_file).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            return safetensors.torch.load_file(checkpoint_file, device="cpu")
        else:
            return torch.load(
                checkpoint_file,
                **torch_load_kwargs,
            )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' " f"at '{checkpoint_file}'. "
            )


def _load_state_dict_into_model(model_to_load, state_dict: OrderedDict) -> List[str]:
    # Convert old format to new format if needed from a PyTorch state_dict
    # copy state_dict so _load_from_state_dict can modify it
    state_dict = state_dict.copy()
    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, prefix: str = ""):
        args = (state_dict, prefix, {}, True, [], [], error_msgs)
        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model_to_load)

    return error_msgs


def _fetch_index_file(
    is_local,
    pretrained_model_name_or_path,
    subfolder,
    use_safetensors,
    cache_dir,
    variant,
    force_download,
    resume_download,
    proxies,
    local_files_only,
    token,
    revision,
    user_agent,
    commit_hash,
):
    if is_local:
        index_file = Path(
            pretrained_model_name_or_path,
            subfolder or "",
            _add_variant(SAFETENSORS_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME, variant),
        )
    else:
        index_file_in_repo = Path(
            subfolder or "",
            _add_variant(SAFETENSORS_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME, variant),
        ).as_posix()
        try:
            index_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=index_file_in_repo,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                commit_hash=commit_hash,
            )
            index_file = Path(index_file)
        except (EntryNotFoundError, EnvironmentError):
            index_file = None

    return index_file
