import os
import cv2
import torch
import warnings
import numpy as np
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
from huggingface_hub import hf_hub_download

# IMPORTANT: this file intentionally imports the single model (RIFE_HDv3).
# It does NOT try multiple model fallbacks
from .RIFE_HDv3 import Model

warnings.filterwarnings("ignore")
# disable gradients for inference
torch.set_grad_enabled(False)


class RIFE():
    """Minimal RIFE inference wrapper.

    This class mirrors the behavior and expectations of the simple image-only
    inference script (the one that imports RIFE_HDv3.Model and calls
    model.load_model(model_dir)). It purposefully does NOT include the
    training/research-specific imports or fallback model loading found in
    the video script.

    Usage (short):
        rife = RIFE(model_dir="train_log")
        outputs = rife.interpolate_image("a.png", "b.png", exp=4)
        rife.interpolate_video("in.mp4", output_path="out.mp4", exp=2)
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: Union[str, os.PathLike], device=None, **kwargs):
        # Prefer local safetensors loading (preserves the same mapping used by Model.load_model)
        path = str(pretrained_model_or_path)
        # Resolve candidate safetensors file
        if os.path.isdir(path):
            candidate = os.path.join(path, "model.safetensors")
        else:
            candidate = path if path.endswith(".safetensors") else os.path.join(path, "model.safetensors")

        if os.path.exists(candidate):
            # use the Model.load_model path which loads into self.flownet correctly
            m = Model()
            try:
                m.device()
            except Exception:
                pass
            model_dir = path if path.endswith(os.sep) else (os.path.dirname(candidate) + os.sep)
            m.load_model(model_dir)
            return cls(model=m, device=device)

        # not local: try to download from HF Hub and load with the same loader
        try:
            print("Attempting to download model.safetensors from Hugging Face Hub...")
            hub_file = hf_hub_download(repo_id=path, filename="model.safetensors", **kwargs)
            if os.path.exists(hub_file):
                m = Model()
                try:
                    m.device()
                except Exception:
                    pass
                model_dir = os.path.dirname(hub_file) + os.sep
                m.load_model(model_dir)
                return cls(model=m, device=device)
        except Exception as e:
            # fall back below if hf_hub_download failed / repo not found / private repo etc.
            print("HF download failed or not a hub id:", e)

        # final fallback: use HF mixin behaviour (for other formats / identifiers)
        print("using the fallback from_pretrained()")
        model = Model.from_pretrained(pretrained_model_or_path, **kwargs)
        return cls(model=model, device=device)


    def __init__(self, model: Union[Model, str], device: Optional[torch.device] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        # so the the model can be loaded in two ways
        # either by the .load_model() function
        if isinstance(model, str):
            print("The model is being loaded using .load_model()")
            self.model = Model()
            self.model.load_model(model)
            print("Loaded v3.x HD model.")
            self.model.eval()
        # or by using 
        else:
            print("The model is being loaded using from_pretrained()")
            self.model = model
            self.model.to(self.device)
            self.model.eval()

    # ----------------------------- helpers ---------------------------------
    def _read_image_tensor(self, path: str) -> Tuple[torch.Tensor, int, int, bool]:
        """Read an image from disk and convert to a torch tensor on self.device.

        Returns (tensor, height, width, is_exr).
        For EXR images we preserve floating point values and don't divide by 255.
        For normal images we convert to float in range [0,1].
        """
        if path.lower().endswith('.exr'):
            img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise FileNotFoundError(path)
            h, w = img.shape[:2]
            tensor = torch.tensor(img.transpose(2, 0, 1)).to(self.device).unsqueeze(0)
            return tensor, h, w, True
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(path)
            h, w = img.shape[:2]
            tensor = torch.tensor(img.transpose(2, 0, 1)).to(self.device).unsqueeze(0).float() / 255.0
            return tensor, h, w, False

    def _pad_pair_to_32(self, img0: torch.Tensor, img1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]:
        """Pad both tensors so height and width are divisible by 32. Returns padding tuple used (left,right,top,bottom) like F.pad expects.
        Both tensors must have same spatial dimensions before padding."""
        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)
        return img0, img1, padding

    def _make_inference(self, I0: torch.Tensor, I1: torch.Tensor, n: int) -> List[torch.Tensor]:
        """Recursively generate n middle frames between I0 and I1.
        n must be >= 1 and should equal (2**exp - 1) when used by callers."""
        middle = self.model.inference(I0, I1)
        if n == 1:
            return [middle]
        first_half = self._make_inference(I0, middle, n=n // 2)
        second_half = self._make_inference(middle, I1, n=n // 2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def _transfer_audio(self, source_video: str, target_video: str):
        """Transfer audio from source video to target video."""
        import shutil
        
        temp_audio_file = "./temp/audio.mkv"
        
        # Create temp directory
        if os.path.isdir("temp"):
            shutil.rmtree("temp")
        os.makedirs("temp")
        
        # Extract audio
        os.system(f'ffmpeg -y -i "{source_video}" -c:a copy -vn {temp_audio_file}')
        
        target_no_audio = os.path.splitext(target_video)[0] + "_noaudio" + os.path.splitext(target_video)[1]
        os.rename(target_video, target_no_audio)
        
        # Merge audio and video
        os.system(f'ffmpeg -y -i "{target_no_audio}" -i {temp_audio_file} -c copy "{target_video}"')
        
        if os.path.getsize(target_video) == 0:
            # Try AAC conversion
            temp_audio_file = "./temp/audio.m4a"
            os.system(f'ffmpeg -y -i "{source_video}" -c:a aac -b:a 160k -vn {temp_audio_file}')
            os.system(f'ffmpeg -y -i "{target_no_audio}" -i {temp_audio_file} -c copy "{target_video}"')
            
            if os.path.getsize(target_video) == 0:
                os.rename(target_no_audio, target_video)
                print("Audio transfer failed. Interpolated video will have no audio")
            else:
                print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
                os.remove(target_no_audio)
        else:
            os.remove(target_no_audio)
        
        # Cleanup
        shutil.rmtree("temp")

    # --------------------------- public API --------------------------------
    def interpolate_image(self, img0_path: str, img1_path: str, exp: int = 4, ratio: float = 0.0,
                          rthreshold: float = 0.02, rmaxcycles: int = 8, output_dir: str = "images/output") -> List[str]:
        """Interpolate between two images and save outputs.

        If ratio is > 0, the method will attempt to return the middle frame that
        corresponds to the given ratio using a bisection-like search with
        rthreshold and rmaxcycles limits. If ratio==0, it generates 2**exp frames
        (including endpoints) by repeatedly inserting mid-frames.

        Returns a list of saved file paths (in chronological order).
        """
        os.makedirs(output_dir, exist_ok=True)

        img0, h0, w0, exr0 = self._read_image_tensor(img0_path)
        img1, h1, w1, exr1 = self._read_image_tensor(img1_path)
        if h0 != h1 or w0 != w1:
            raise ValueError("Input images must have the same dimensions")

        img0, img1, padding = self._pad_pair_to_32(img0, img1)
        h, w = h0, w0

        if ratio:
            img_list = [img0]
            img0_ratio = 0.0
            img1_ratio = 1.0
            if ratio <= img0_ratio + rthreshold / 2:
                middle = img0
            elif ratio >= img1_ratio - rthreshold / 2:
                middle = img1
            else:
                tmp_img0 = img0
                tmp_img1 = img1
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
            img_list.extend([middle, img1])
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

        saved = []
        for i, t in enumerate(img_list):
            out_name = os.path.join(output_dir, f"img{i}.{'exr' if exr0 and exr1 else 'png'}")
            if exr0 and exr1:
                cv2.imwrite(out_name, (t[0]).cpu().numpy().transpose(1, 2, 0)[:h, :w],
                            [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            else:
                cv2.imwrite(out_name, (t[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
            saved.append(out_name)

        return saved

    def interpolate_video(self, video_path: str, output_path: Optional[str] = None, 
                          exp: int = 1, fps: Optional[float] = None,
                          scale: float = 1.0, montage: bool = False,
                          ext: str = 'mp4', transfer_audio: bool = True) -> str:
        """Interpolate frames in a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (auto-generated if None)
            exp: Exponential factor for frame multiplication (multi = 2**exp)
            fps: Target FPS (auto-calculated if None)
            scale: Scaling factor for processing (0.25, 0.5, 1.0, 2.0, 4.0)
            montage: Whether to create side-by-side comparison
            ext: Output video extension
            transfer_audio: Whether to transfer audio from source
            
        Returns:
            Path to the output video file
        """
        #TODO: these are some external dependencies we should probably manage better
        # this is meant to be a standalone script so that user can run it within the
        # huggingface ecosystem without installing a lot of extra stuff, manage these
        # dependencies better later
        import skvideo.io
        import shutil
        import _thread
        from queue import Queue
        from tqdm import tqdm
        from .pytorch_msssim import ssim_matlab
        
        multi = 2 ** exp
        assert scale in [0.25, 0.5, 1.0, 2.0, 4.0]
        
        # Read video properties
        videoCapture = cv2.VideoCapture(video_path)
        source_fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        videoCapture.release()
        
        if fps is None:
            fps_not_assigned = True
            fps = source_fps * multi
        else:
            fps_not_assigned = False
        
        # Setup video reader
        videogen = skvideo.io.vreader(video_path)
        print("Video reading started...")
        lastframe = next(videogen)
        print("Video reading completed.", flush=True)
        h, w, _ = lastframe.shape
        
        # Generate output path
        video_path_wo_ext, _ = os.path.splitext(video_path)
        if output_path is None:
            output_path = f'{video_path_wo_ext}_{multi}X_{int(np.round(fps))}fps.{ext}'
        
        print(f'{video_path_wo_ext}.{ext}, {tot_frame} frames in total, {source_fps}FPS to {fps}FPS')
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vid_out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Helper functions for threading
        def clear_write_buffer(write_buffer):
            while True:
                item = write_buffer.get()
                if item is None:
                    break
                vid_out.write(item[:, :, ::-1])
        
        def build_read_buffer(read_buffer, videogen):
            try:
                for frame in videogen:
                    if montage:
                        frame = frame[:, left: left + w]
                    read_buffer.put(frame)
            except:
                pass
            read_buffer.put(None)
        
        # Setup montage if needed
        if montage:
            left = w // 4
            w = w // 2
            lastframe = lastframe[:, left: left + w]
        
        # Calculate padding
        tmp = max(128, int(128 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        
        # Setup buffers and threads
        write_buffer = Queue(maxsize=500)
        read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
        _thread.start_new_thread(clear_write_buffer, (write_buffer,))
        
        # Process frames
        I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = F.pad(I1, padding)
        temp = None
        
        pbar = tqdm(total=tot_frame)
        
        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                frame = read_buffer.get()
            
            if frame is None:
                break
            
            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = F.pad(I1, padding)
            
            # Calculate SSIM for static frame detection
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            
            break_flag = False
            if ssim > 0.996:
                frame = read_buffer.get()
                if frame is None:
                    break_flag = True
                    frame = lastframe
                else:
                    temp = frame
                I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
                I1 = F.pad(I1, padding)
                I1 = self.model.inference(I0, I1, scale=scale)
                I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            
            # Generate interpolated frames
            if ssim < 0.2:
                output = [I0] * (multi - 1)
            else:
                output = self._make_inference(I0, I1, multi - 1)
            
            # Write frames
            if montage:
                write_buffer.put(np.concatenate((lastframe, lastframe), 1))
                for mid in output:
                    mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
                    write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
            else:
                write_buffer.put(lastframe)
                for mid in output:
                    mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
                    write_buffer.put(mid[:h, :w])
            
            pbar.update(1)
            lastframe = frame
            
            if break_flag:
                break
        
        # Write final frame
        if montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        else:
            write_buffer.put(lastframe)
        write_buffer.put(None)
        
        # Wait for write buffer to empty
        import time
        while not write_buffer.empty():
            time.sleep(0.1)
        
        pbar.close()
        vid_out.release()
        
        # Transfer audio if appropriate
        if transfer_audio and fps_not_assigned:
            try:
                self._transfer_audio(video_path, output_path)
            except:
                print("Audio transfer failed. Interpolated video will have no audio")
        
        return output_path