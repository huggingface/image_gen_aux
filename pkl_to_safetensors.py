# from safetensors.torch import save_file
# import torch

# pkl_weights = torch.load("flownet.pkl", map_location="cpu")
# save_file(pkl_weights, "flownet.safetensors")


# requires: safetensors, huggingface_hub
import hashlib, os
from huggingface_hub import hf_hub_download

hub_path = hf_hub_download("1himan/RIFE", "model.safetensors")  # adapt repo/filename
local_path = "model.safetensors"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

print("local sha256:", sha256(local_path))
print("hub   sha256:", sha256(hub_path))
print("sizes      :", os.path.getsize(local_path), os.path.getsize(hub_path))


from safetensors.torch import load_file
import torch

local_state = load_file(local_path)
hub_state   = load_file(hub_path)

# quick checks
print("keys equal:", set(local_state.keys()) == set(hub_state.keys()))
# compute max abs diff across all tensors
max_diff = 0.0
for k in local_state.keys():
    a = local_state[k]
    b = hub_state[k]
    a_t = a if isinstance(a, torch.Tensor) else torch.tensor(a)
    b_t = b if isinstance(b, torch.Tensor) else torch.tensor(b)
    if a_t.shape != b_t.shape:
        print("shape mismatch for", k, a_t.shape, b_t.shape)
    diff = torch.max((a_t - b_t).abs()).item()
    if diff > max_diff:
        max_diff = diff
print("max absolute difference:", max_diff)