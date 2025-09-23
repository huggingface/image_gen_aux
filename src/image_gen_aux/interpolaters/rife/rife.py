import os
import cv2
import torch
from torch.nn import functional as F
import numpy as np
import PIL.Image
from .RIFE_HDv3 import Model
from ..interpolater import Interpolater
from huggingface_hub.utils import validate_hf_hub_args


class RIFE(Interpolater):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # load RIFE model once
        self.model = Model()
        self.model.load_model(model_dir, -1)
        self.model.eval()
        self.model.device()
        print("Loaded v3.x HD model.")
    
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        model = Model.from_pretrained(pretrained_model_or_path, **kwargs)
        return cls(model)

    def interpolate(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        """Abstract method (required by base class). Not used here directly."""
        raise NotImplementedError("Use interpolate_images() instead")

    def interpolate_images(self, img0_path, img1_path, exp=4, ratio=0.0, rthreshold=0.02, rmaxcycles=8, save_output=True):
        """Interpolate between two images using RIFE.

        Args:
            img0_path (str): path to first image
            img1_path (str): path to second image
            exp (int): number of interpolation iterations (ignored if ratio > 0)
            ratio (float): specific interpolation ratio (0–1). If >0, ignores exp
            rthreshold (float): threshold for ratio matching
            rmaxcycles (int): max number of recursive cycles
            save_output (bool): whether to save results in ./output

        Returns:
            List of numpy arrays of interpolated frames.
        """

        # load images
        if img0_path.endswith('.exr') and img1_path.endswith('.exr'):
            img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(self.device)).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device)).unsqueeze(0)
        else:
            img0 = cv2.imread(img0_path, cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)

        # pad
        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        # interpolation
        if ratio > 0.0:
            img_list = [img0]
            img0_ratio, img1_ratio = 0.0, 1.0
            if ratio <= img0_ratio + rthreshold / 2:
                middle = img0
            elif ratio >= img1_ratio - rthreshold / 2:
                middle = img1
            else:
                tmp_img0, tmp_img1 = img0, img1
                for _ in range(rmaxcycles):
                    middle = self.model.inference(tmp_img0, tmp_img1)
                    middle_ratio = (img0_ratio + img1_ratio) / 2
                    if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                        break
                    if ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1)
        else:
            img_list = [img0, img1]
            for _ in range(exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = self.model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

        # save results
        results = []
        if save_output:
            if not os.path.exists('output'):
                os.mkdir('output')

        for i, tensor_img in enumerate(img_list):
            np_img = (tensor_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            results.append(np_img)
            if save_output:
                cv2.imwrite(f'output/img{i}.png', np_img)

        return results
