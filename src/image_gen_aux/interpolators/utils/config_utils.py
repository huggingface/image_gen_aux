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

import copy
import json
import os
from typing import Optional, Union

import numpy as np

from ...utils.logging import get_logger
from . import __version__


logger = get_logger(__name__)


class ModelConfig:
    def __init__(self):
        super().__setattr__("_internal_dict", {})

    def __getattr__(self, name):
        if name not in self._internal_dict:
            raise AttributeError(f"Config does not have attribute: {name}")
        return self._internal_dict.get(name)

    def __setattr__(self, name, value):
        self._internal_dict[name] = value

    def __getitem__(self, name):
        if name not in self._internal_dict:
            raise KeyError(f"Config does not have key: {name}")
        return self._internal_dict.get(name)

    def __setitem__(self, name, value):
        self._internal_dict[name] = value

    def __delitem__(self, name):
        del self._internal_dict[name]

    def __iter__(self):
        return iter(self._internal_dict)

    def __len__(self):
        return len(self._internal_dict)

    def __repr__(self):
        return repr(self._internal_dict)

    def __str__(self):
        return str(self._internal_dict)

    def __contains__(self, name):
        return name in self._internal_dict

    @staticmethod
    def _to_json_saveable(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return value

    def to_json_string(self, class_name: Optional[str] = None) -> str:
        r"""
        Serialize the configuration to a JSON formatted string.

        Returns:
            str:
                String containing all the attributes that make up the configuration instance in JSON format.
        """
        config_dict: dict = copy.deepcopy(self._internal_dict)
        config_dict["_class_name"] = class_name or self.__class__.__name__
        config_dict["_backbones_version"] = __version__

        config_dict = {k: self._to_json_saveable(v) for k, v in config_dict.items()}
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, path: Union[str, os.PathLike], class_name: Optional[str] = None) -> None:
        r"""
        Serialize the configuration to a JSON formatted file.

        Args:
            path (str or os.PathLike):
                The path to the file where the configuration will be saved.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json_string(class_name))


def is_primitive_type(value) -> bool:
    return isinstance(value, (int, float, str, bool))


def is_primitive_type_dict(value) -> bool:
    return isinstance(value, dict) and all(is_primitive_type(v) for v in value.values())


def is_primitive_type_list(value) -> bool:
    return isinstance(value, (list, tuple)) and all(is_primitive_type(v) for v in value)
