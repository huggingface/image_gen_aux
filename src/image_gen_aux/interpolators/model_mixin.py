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

import inspect
import itertools
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import safetensors.torch
import torch
import torch.nn as nn
from decorator import decorator
from huggingface_hub import create_repo, split_torch_state_dict_into_shards

from ..utils.constants import (
    CONFIG_NAME,
    SAFETENSORS_WEIGHTS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from ..utils.logging import get_logger
from .utils.config_utils import ModelConfig, is_primitive_type, is_primitive_type_dict, is_primitive_type_list
from .utils.hub_utils import (
    PushToHubMixin,
    _add_variant,
    _get_checkpoint_shard_files,
    _get_model_file,
    extract_commit_hash,
    http_user_agent,
    load_or_create_model_card,
    populate_model_card,
)
from .utils.model_loading_utils import _fetch_index_file, _load_state_dict_into_model, load_state_dict


__version__ = "0.0.1"
logger = get_logger(__name__)

_REGEX_SHARD = re.compile(r"(.*?)-\d{5}-of-\d{5}")


# TODO: Maintain common modeling mixin file and move from interpolators to image_gen_aux/utils


def get_parameter_device(parameter: torch.nn.Module) -> torch.device:
    """
    Gets the device of a PyTorch module's parameters or buffers.

    Args:
        parameter (`torch.nn.Module`): The PyTorch module from which to get the device.

    Returns:
        `torch.device`: The device of the module's parameters or buffers.
    """
    try:
        parameters_and_buffers = itertools.chain(parameter.parameters(), parameter.buffers())
        return next(parameters_and_buffers).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5
        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module) -> torch.dtype:
    """
    Gets the data type of a PyTorch module's parameters or buffers.

    Args:
        parameter (`torch.nn.Module`): The PyTorch module from which to get the data type.

    Returns:
        `torch.dtype`: The data type of the module's parameters or buffers.
    """
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5
        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class ModelMixin(nn.Module, PushToHubMixin):
    config_filename: str = CONFIG_NAME
    _automatically_saved_args: List[str] = ["_image_gen_aux_version", "_class_name", "_name_or_path"]
    config: ModelConfig

    def __init__(self) -> None:
        super().__init__()

        if not hasattr(self, "config"):
            self.config = ModelConfig()
            logger.warning(
                f"Model {self.__class__.__name__} does not have a `config` attribute. Creating an empty one. Did you miss a `register_to_config` decorator on the `__init__` method?"
            )

    def register_to_config(self, **kwargs: Dict[str, Any]) -> "ModelMixin":
        for key, value in kwargs.items():
            if not (is_primitive_type(value) or is_primitive_type_list(value) or is_primitive_type_dict(value)):
                raise TypeError(
                    f"Invalid type for {key}: {type(value)}. Must be a primitive type or a list/dict of primitive types."
                )
            self.config[key] = value
        return self

    def init_weights(self, module: Any) -> nn.Module:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        return self

    def save_config(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        r"""
        Save the model configuration to a directory so that it can be reloaded using the `from_config` class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory to which the configuration file will be saved. Will be created if it does not exist.
            push_to_hub (`bool`, defaults to `False`):
                Whether or not to push the configuration to the 🤗 Hugging Face Hub after saving.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided `save_directory` ({save_directory}) should be a directory, not a file.")

        os.makedirs(save_directory, exist_ok=True)

        output_config_file = os.path.join(save_directory, self.config_filename)
        self.config.to_json_file(output_config_file, self.__class__.__name__)
        logger.info(f"Configuration saved in {output_config_file}.")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", False)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

            self._upload_folder(
                save_directory, repo_id, token=token, commit_message=commit_message, create_pr=create_pr
            )

    @classmethod
    def load_config(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        return_unused_kwargs: bool = False,
        return_commit_hash: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], ...]:
        _ = kwargs.pop("mirror", None)
        subfolder = kwargs.pop("subfolder", None)
        user_agent = kwargs.pop("user_agent", {})

        user_agent = {**user_agent, "file_type": "config"}
        user_agent = http_user_agent(user_agent)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            folder_config_name_or_path = os.path.join(pretrained_model_name_or_path, cls.config_filename)
            if subfolder is not None and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, cls.config_filename)
            ):
                config_file = os.path.join(pretrained_model_name_or_path, subfolder, cls.config_filename)
            elif os.path.isfile(folder_config_name_or_path):
                config_file = folder_config_name_or_path
            else:
                raise EnvironmentError(
                    f"Configuration file {cls.config_filename} not found in directory {pretrained_model_name_or_path}."
                )
        else:
            # TODO: Implement loading from Hugging Face Hub
            raise NotImplementedError("Loading from Hugging Face Hub is not yet implemented.")

        try:
            config_dict = cls._dict_from_json_file(config_file)
            commit_hash = extract_commit_hash(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"Configuration file {config_file} is not a valid JSON file.")

        if not return_unused_kwargs and not return_commit_hash:
            return config_dict

        outputs = (config_dict,)
        if return_unused_kwargs:
            outputs += (kwargs,)
        if return_commit_hash:
            outputs += (commit_hash,)
        return outputs

    @classmethod
    def from_config(
        cls, config: Union[ModelConfig, Dict[str, Any]], return_unused_kwargs: bool = False, **kwargs: Dict[str, Any]
    ) -> "ModelMixin":
        if not isinstance(config, ModelConfig) and not isinstance(config, dict):
            raise ValueError("`config` must be an instance of `ModelConfig` or a dictionary")
        if isinstance(config, ModelConfig):
            config = config._internal_dict

        init_dict, unused_kwargs, hidden_dict = cls.extract_init_dict(config, **kwargs)

        # Allow dtype to be specified on initialization
        if "dtype" in unused_kwargs:
            init_dict["dtype"] = unused_kwargs.pop("dtype")

        model = cls(**init_dict)
        model.register_to_config(**hidden_dict)

        unused_kwargs = {**unused_kwargs, **kwargs}
        if return_unused_kwargs:
            return model, unused_kwargs
        return model

    @classmethod
    def extract_init_dict(cls, config_dict: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], ...]:
        # 1. Copy original config dict
        original_dict = dict(config_dict.items())

        # 2. Retrieve expected attributes from __init__ signature
        expected_keys = cls._get_init_keys(cls)
        expected_keys.remove("self")

        # Remove private attributes
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}

        # 3. Create keyword arguments for __init__
        init_dict = {}
        for key in expected_keys:
            # if key in both kwargs and config_dict, we overwrite the value in config_dict
            if key in kwargs and key in config_dict:
                config_dict[key] = kwargs.pop(key)
            elif key in kwargs:
                init_dict[key] = kwargs.pop(key)
            elif key in config_dict:
                init_dict[key] = config_dict.pop(key)

        if len(config_dict) > 0:
            logger.warning(
                f"The config attributes {config_dict} were passed to {cls.__name__}, "
                f"but are not expected and will be ignored. Please verify your "
                f"{cls.config_filename} configration file."
            )

        passed_keys = set(init_dict.keys())
        if len(expected_keys - passed_keys) > 0:
            logger.info(f"{expected_keys - passed_keys} was not found in config. Values will be set to default.")

        unused_kwargs = {**config_dict, **kwargs}
        hidden_config_dict = {k: v for k, v in original_dict.items() if k not in init_dict}

        return init_dict, unused_kwargs, hidden_config_dict

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "10GB",
        push_to_hub: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        r"""
        Save a model and its configuration file to a directory so that it can be reloaded using the
        `from_pretrained` class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory to which the model and configuration files will be saved. Will be created if it does not exist.
            is_main_process (`bool`, defaults to `True`):
                Whether or not this is the main process. Only the main process should save the model to avoid race conditions
                in a distributed setting.
            safe_serialization (`bool`, defaults to `True`):
                Whether or not to convert the model weights to the `safetensors` format. If `False`, the model weights will be
                saved in the `torch` format.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `<weights_name>.<variant>.<extension>`.
            max_shard_size (`int` or `str`, defaults to `"10GB"`):
                Maximum size of each shard. If a model exceeds this size, it is split into multiple shards. If expressed as
                a string, it needs to be digits followed by the unit (e.g., `"10GB"`). If expressed as an integer, it must
                be in bytes.
            push_to_hub (`bool`, defaults to `False`):
                Whether or not to push the model to the 🤗 Hugging Face Hub after saving.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided `save_directory` ({save_directory}) should be a directory, not a file.")

        weights_name = SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        weights_name = _add_variant(weights_name, variant)
        weights_name_split = weights_name.split(".")

        if len(weights_name_split) in [2, 3]:
            weights_name_pattern = weights_name_split[0] + "{suffix}." + ".".join(weights_name_split[1:])
        else:
            raise ValueError(f"Invalid weights name `{weights_name}` provided")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", False)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        # Save the config
        self.save_config(save_directory)

        # Save the model
        state_dict = self.state_dict()
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, max_shard_size=max_shard_size, filename_pattern=weights_name_pattern
        )

        if is_main_process:
            for filename in os.listdir(save_directory):
                if filename in state_dict_split.filename_to_tensors.keys():
                    continue
                full_filename = os.path.join(save_directory, filename)
                if not os.path.isfile(full_filename):
                    continue
                weights_without_ext = (
                    weights_name_pattern.replace(".bin", "").replace(".safetensors", "").replace("{suffix}", "")
                )
                filename_without_ext = filename.replace(".bin", "").replace(".safetensors", "")

                # Make sure that file to be deleted matches format of sharded file
                if (
                    filename.startswith(weights_without_ext)
                    and _REGEX_SHARD.fullmatch(filename_without_ext) is not None
                ):
                    os.remove(full_filename)

        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            filepath = os.path.join(save_directory, filename)

            if safe_serialization:
                safetensors.torch.save_file(shard, filepath, metadata={"format": "pt"})
            else:
                torch.save(shard, filepath)

        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFETENSORS_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = _add_variant(save_index_file, variant)

            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)

            logger.info(
                f"The model size is bigger that the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split into {len(state_dict_split.filename_to_tensors)} shards. You can find where each parameter has "
                f"been saved in the index located at {save_index_file}."
            )
        else:
            path_to_weights = os.path.join(save_directory, weights_name)
            logger.info(f"Model weights saved in {path_to_weights}")

        if push_to_hub:
            model_card = load_or_create_model_card(repo_id, token=token)
            model_card = populate_model_card(model_card)
            model_card.save(Path(save_directory, "README.md").as_posix())
            self._upload_folder(
                save_directory, repo_id, token=token, commit_message=commit_message, create_pr=create_pr
            )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs: Dict[str, Any]
    ) -> "ModelMixin":
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "_image_gen_aux_version": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        config, unused_kwargs, commit_hash = cls.load_config(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )

        # Determine if we're loading from a directory of sharded checkpoints
        is_sharded = False
        index_file = None
        is_local = os.path.isdir(pretrained_model_name_or_path)
        index_file = _fetch_index_file(
            is_local=is_local,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder or "",
            use_safetensors=use_safetensors,
            cache_dir=cache_dir,
            variant=variant,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            user_agent=user_agent,
            commit_hash=commit_hash,
        )
        if index_file is not None and index_file.is_file():
            is_sharded = True

        # Load model
        model_file = None
        if is_sharded:
            sharded_ckpt_cached_folder, sharded_metadata = _get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                index_file,
                cache_dir=cache_dir,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder or "",
            )
        elif use_safetensors and not is_sharded:
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
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
            except IOError as e:
                logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                if not allow_pickle:
                    raise
                logger.warning(
                    "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                )

        if model_file is None and not is_sharded:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=_add_variant(WEIGHTS_NAME, variant),
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

        model = cls.from_config(config, **unused_kwargs)
        state_dict = load_state_dict(model_file)

        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": error_msgs,
        }

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.eval()

        if output_loading_info:
            return model, loading_info
        return model

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

    @staticmethod
    def _get_init_keys(input_cls):
        return set(dict(inspect.signature(input_cls.__init__).parameters).keys())

    @classmethod
    def _dict_from_json_file(cls, file_path: Union[str, os.PathLike]) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = f.read()
        return json.loads(config_dict)

    @classmethod
    def _load_pretrained_model(
        cls,
        model: "ModelMixin",
        state_dict: OrderedDict,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        ignore_mismatched_sizes: bool = False,
    ) -> Tuple["ModelMixin", List[str], List[str], List[str], List[str]]:
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        loaded_keys = list(state_dict.keys())
        expected_keys = list(model_state_dict.keys())
        original_loaded_keys = loaded_keys
        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Make sure we are able to load base models as well as derived models (with heads)
        model_to_load = model

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}"
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at "
                f"{pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably "
                f"train this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at "
                f"{pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the "
                f"checkpoint was trained on, you can already use {model.__class__.__name__} for predictions "
                f"without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at "
                f"{pretrained_model_name_or_path} and are newly initialized because the shapes did not "
                f"match:\n{mismatched_warning}\nYou should probably train this model on a down-stream task to be "
                f"able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    @property
    def config_repr(self):
        return f"{self.__class__.__name__}({self.config.to_json_string()[:-1]})"


@decorator
def register_to_config(function, *args, **kwargs):
    def wrapper(func):
        nonlocal args, kwargs

        sig = inspect.signature(func)
        defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
        self, args = args[0], args[1:]

        if not isinstance(self, ModelMixin):
            raise ValueError(
                "`register_to_config` decorator can only be used on the `__init__` method of classes that inherit from `ModelMixin`"
            )

        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        config = {
            **defaults,
            **dict(zip(list(sig.parameters.keys())[1:], bound.args[1:])),
            **bound.kwargs,
        }

        if not hasattr(self, "config"):
            self.config = ModelConfig()

        for key, value in config.items():
            if is_primitive_type(value) or is_primitive_type_list(value) or is_primitive_type_dict(value):
                self.config[key] = value

        func(self, *args, **kwargs)

    if function is None:
        return wrapper
    return wrapper(function)
