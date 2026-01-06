import glob
import os

from huggingface_hub import hf_hub_download, model_info
from huggingface_hub.errors import HfHubHTTPError


def get_model_path(pretrained_model_or_path, filename=None, subfolder=None):
    """
    Retrieves the path to the model file.

    If `pretrained_model_or_path` is a file, it returns the path directly.
    Otherwise, it attempts to find a `.safetensors` file associated with the given model path.
    If no `.safetensors` file is found, it raises a `FileNotFoundError`.

    Parameters:
    - pretrained_model_or_path (str): Path to the pretrained model or directory containing the model.
    - filename (str, optional): Specific filename to load. If not provided, the function will search for a `.safetensors` file.
    - subfolder (str, optional): Subfolder within the model directory to look for the file.

    Returns:
    - str: Path to the model file.

    Raises:
    - FileNotFoundError: If no `.safetensors` file is found when `filename` is not provided.
    """
    if os.path.isfile(pretrained_model_or_path):
        return pretrained_model_or_path

    model_dir = os.path.abspath(os.path.join("models", pretrained_model_or_path))
    if os.path.isdir(model_dir):
        safetensor_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
        if safetensor_files:
            if len(safetensor_files) > 1:
                print(f"Warning: Multiple .safetensors files found in {model_dir}, using {safetensor_files[0]}")
            return safetensor_files[0]

    if filename is None:
        # Try to find a safetensors file on Hugging Face
        try:
            info = model_info(pretrained_model_or_path)
        except HfHubHTTPError as e:
            raise FileNotFoundError(f"Model '{pretrained_model_or_path}' not found on Hugging Face Hub.") from e

        # If the filename is not passed, we only try to load a safetensor
        filename = next(
            (sibling.rfilename for sibling in info.siblings if sibling.rfilename.endswith(".safetensors")), None
        )
        if filename is None:
            raise FileNotFoundError(f"No .safetensors checkpoint found for model: {pretrained_model_or_path}")

    return hf_hub_download(pretrained_model_or_path, filename, subfolder=subfolder, local_dir=model_dir)
