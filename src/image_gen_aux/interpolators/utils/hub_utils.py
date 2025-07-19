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

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import uuid4

import torch
from huggingface_hub import (
    ModelCard,
    ModelCardData,
    create_repo,
    hf_hub_download,
    model_info,
    snapshot_download,
    upload_folder,
)
from huggingface_hub.constants import HF_HUB_DISABLE_TELEMETRY, HF_HUB_OFFLINE
from huggingface_hub.file_download import REGEX_COMMIT_HASH
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    is_jinja_available,
)
from requests import HTTPError

from ...utils.constants import HUGGINGFACE_CO_RESOLVE_ENDPOINT
from ...utils.logging import get_logger
from . import __version__


logger = get_logger(__name__)

MODEL_CARD_TEMPLATE_PATH = Path(__file__).parent / "model_card_template.md"
SESSION_ID = uuid4().hex


class PushToHubMixin:
    """
    Mixin to push a model to the 🤗 Hugging Face Hub.
    """

    def _upload_folder(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all files in `working_dir` to `repo_id`.
        """
        if commit_message is None:
            commit_message = f"Upload {self.__class__.__name__}"

        logger.info(f"Uploading the files of {working_dir} to {repo_id}.")
        return upload_folder(
            repo_id=repo_id, folder_path=working_dir, token=token, commit_message=commit_message, create_pr=create_pr
        )

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[str] = None,
        create_pr: bool = False,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ) -> str:
        """
        Upload model to the 🤗 Hugging Face Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your model files to. It should contain your organization
                name when pushing to an organization. `repo_id` can also be a path to a local directory.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Default to `"Upload {class_name}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. The token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to convert the model weights to the `safetensors` format.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
        """
        repo_id = create_repo(repo_id, private=private, token=token, exist_ok=True).repo_id

        # Save all files.
        save_kwargs = {"safe_serialization": safe_serialization}
        save_kwargs.update({"variant": variant})

        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir, **save_kwargs)

            return self._upload_folder(
                tmpdir,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )


def load_or_create_model_card(
    repo_id_or_path: Optional[str] = None,
    token: Optional[str] = None,
    from_training: bool = False,
    model_description: Optional[str] = None,
    base_model: Optional[str] = None,
    license: Optional[str] = None,
) -> ModelCard:
    r"""
    Loads or creates a model card.

    Args:
        repo_id_or_path (`str`, *optional*):
            The repository ID or local path where to look for the model card. If `None`, a new model card is created.
        token (`str`, *optional*):
            Authentication token. Will default to the stored token. See https://huggingface.co/settings/token for more
            details.
        from_training (`bool`, *optional*):
            Whether or not this model was created from a training script. Defaults to `False`.
        model_description (`str`, *optional*):
            The description of the model card. If `None`, a default description is used.
        base_model (`str`, *optional*):
            The name of the base model.
        license (`str`, *optional*):
            License of the output artifact.

    Returns:
        ModelCard:
            The loaded/created model card.
    """
    if not is_jinja_available():
        raise ValueError(
            "Modelcard rendering is based on Jinja templates. Please make sure to have `jinja` installed "
            "before using `load_or_create_model_card`. To install it, please run `pip install Jinja2`."
        )

    try:
        # Check if the model card is present on the remote repo
        model_card = ModelCard.load(repo_id_or_path, token=token)
    except (EntryNotFoundError, RepositoryNotFoundError):
        # Create a new model card from template
        if from_training:
            model_card = ModelCard.from_template(
                card_data=ModelCardData(
                    license=license,
                    library_name="backbones",
                    base_model=base_model,
                ),
                template_path=MODEL_CARD_TEMPLATE_PATH,
                model_description=model_description,
            )
        else:
            card_data = ModelCardData()
            model_description = "This is the model card of a 🦴 backbones model that has been pushed to the Hub. This model card has been automatically generated."
            model_card = ModelCard.from_template(card_data, model_description=model_description)

    return model_card


def populate_model_card(model_card: ModelCard, tags: Optional[Union[str, List[str]]] = None) -> ModelCard:
    r"""Populates the `model_card` with library name and optionla `tags`."""
    if model_card.card_data.library_name is None:
        model_card.card_data.library_name = "backbones"

    if tags is not None:
        if isinstance(tags, str):
            tags = [tags]
        if model_card.data.tags is None:
            model_card.data.tags = []
        for tag in tags:
            if tag in model_card.data.tags:
                continue
            model_card.data.tags.append(tag)

    return model_card


def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str] = None):
    r"""Extracts the commit hash from a resolved filename."""
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    r"""Formats a user-agent string with basic info about a request."""
    ua = f"backbones/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if HF_HUB_DISABLE_TELEMETRY or HF_HUB_OFFLINE:
        return ua + "; telemetry/off"
    ua += f"; torch/{torch.__version__}"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def _get_model_file(
    pretrained_model_name_or_path: Union[str, Path],
    *,
    weights_name: str,
    subfolder: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    resume_download: Optional[bool] = None,
    local_files_only: bool = False,
    token: Optional[str] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    revision: Optional[str] = None,
    commit_hash: Optional[str] = None,
):
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    elif os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
            # Load from a PyTorch checkpoint
            model_file = os.path.join(pretrained_model_name_or_path, weights_name)
            return model_file
        elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        ):
            model_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
            return model_file
        else:
            raise EnvironmentError(
                f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}."
            )
    else:
        try:
            # Load model file
            model_file = hf_hub_download(
                pretrained_model_name_or_path,
                filename=weights_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                subfolder=subfolder,
                revision=revision or commit_hash,
            )
            return model_file
        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                "token having permission to this repo with `token` or log in with `huggingface-cli login`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                "this model name. Check the model page at "
                f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
            )
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {weights_name}."
            )
        except HTTPError as err:
            raise EnvironmentError(
                f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n{err}"
            )
        except ValueError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it "
                f"in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a "
                f"directory containing a file named {weights_name}. Set `export HF_HUB_OFFLINE=True` to work in "
                "offline mode."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a file named {weights_name}"
            )


def _check_if_shards_exist_locally(local_dir: str, subfolder: str, original_shard_filenames: List[str]) -> None:
    shards_path = os.path.join(local_dir, subfolder)
    shard_filenames = [os.path.join(shards_path, f) for f in original_shard_filenames]
    for shard_file in shard_filenames:
        if not os.path.exists(shard_file):
            raise ValueError(
                f"{shards_path} does not appear to have a file named {shard_file} which is "
                "required according to the checkpoint index."
            )


def _get_checkpoint_shard_files(
    pretrained_model_name_or_path: Union[str, Path],
    index_filename: Union[str, os.PathLike],
    cache_dir: Optional[str] = None,
    proxies: Optional[Dict] = None,
    resume_download: bool = False,
    local_files_only: bool = False,
    token: Optional[str] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    revision: Optional[str] = None,
    subfolder: str = "",
):
    r"""
    Utility function that performs the following steps:
      - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a valid model_id
        on the Hub
      - returns the list of paths to all the shards, as well as some metadata.
    """
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    original_shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()
    shards_path = os.path.join(pretrained_model_name_or_path, subfolder)

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        _check_if_shards_exist_locally(
            pretrained_model_name_or_path, subfolder=subfolder, original_shard_filenames=original_shard_filenames
        )
        return pretrained_model_name_or_path, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    allow_patterns = original_shard_filenames
    ignore_patterns = ["*.json", "*.md"]
    if not local_files_only:
        # `model_info` call must guarded with the above condition.
        model_files_info = model_info(pretrained_model_name_or_path)
        for shard_file in original_shard_filenames:
            shard_file_present = any(shard_file in k.rfilename for k in model_files_info.siblings)
            if not shard_file_present:
                raise EnvironmentError(
                    f"{shards_path} does not appear to have a file named {shard_file} which is "
                    "required according to the checkpoint index."
                )

    try:
        # Load from URL
        cached_folder = snapshot_download(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            user_agent=user_agent,
        )

    # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
    # we don't have to catch them here. We have also dealt with EntryNotFoundError.
    except HTTPError as e:
        raise EnvironmentError(
            f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {pretrained_model_name_or_path}. You should try"
            " again after checking your internet connection."
        ) from e

    # If `local_files_only=True`, `cached_folder` may not contain all the shard files.
    if local_files_only:
        _check_if_shards_exist_locally(
            local_dir=cache_dir, subfolder=subfolder, original_shard_filenames=original_shard_filenames
        )

    return cached_folder, sharded_metadata


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)
    return weights_name
