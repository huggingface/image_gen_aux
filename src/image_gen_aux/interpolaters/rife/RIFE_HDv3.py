import torch
import torch.nn as nn
from .IFNet_HDv3 import IFNet
from safetensors.torch import load_file
from huggingface_hub import PyTorchModelHubMixin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.flownet = IFNet()
        self.device()
        self.version = 4.25

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    # this is the actual loading function used to load the model weights 
    def load_model(self, path):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        param_dict = load_file(f"{path}model.safetensors")
        self.flownet.load_state_dict(convert(param_dict), strict=False)
        
    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]