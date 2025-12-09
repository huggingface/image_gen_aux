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
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: Union[str, os.PathLike], device=None, **kwargs):
        model= Model.from_pretrained(pretrained_model_or_path, **kwargs)
        return cls(model, device=device)

    def __init__(self, model: Union[Model, str], device: Optional[torch.device] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        self.model = model.to(self.device)
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
                # print("Audio transfer failed. Interpolated video will have no audio")
            else:
                # print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
                os.remove(target_no_audio)
        else:
            os.remove(target_no_audio)
        
        # Cleanup
        shutil.rmtree("temp")

    # --------------------------- public API --------------------------------
    def interpolate_images(self, img0_path: str, img1_path: str, exp: int = 4, ratio: float = 0.0,
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
        """Interpolate frames in a video file using OpenCV only (no sk-video)."""
        try:
            import _thread
            from queue import Queue
            from tqdm import tqdm
            from .pytorch_msssim import ssim_matlab
        except ImportError as e:
            raise ImportError("tqdm and pytorch_msssim are required for video interpolation.") from e

        multi = 2 ** exp
        assert scale in [0.25, 0.5, 1.0, 2.0, 4.0]

        # Read video properties
        videoCapture = cv2.VideoCapture(video_path)
        if not videoCapture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        source_fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        success, lastframe = videoCapture.read()
        if not success:
            raise ValueError(f"Failed to read first frame of {video_path}")
        h, w, _ = lastframe.shape

        if fps is None:
            fps_not_assigned = True
            fps = source_fps * multi
        else:
            fps_not_assigned = False

        video_path_wo_ext, _ = os.path.splitext(video_path)
        if output_path is None:
            output_path = f'{video_path_wo_ext}_{multi}X_{int(np.round(fps))}fps.{ext}'

        # print(f'{video_path_wo_ext}.{ext}, {tot_frame} frames total, {source_fps:.2f}FPS → {fps:.2f}FPS')

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Helper: generator to yield frames from cv2
        def cv2_frame_generator(path):
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()

        # Helper threads
        def clear_write_buffer(write_buffer):
            while True:
                item = write_buffer.get()
                if item is None:
                    break
                vid_out.write(item)

        def build_read_buffer(read_buffer, generator):
            try:
                for frame in generator:
                    if montage:
                        frame = frame[:, left:left + w]
                    read_buffer.put(frame)
            except:
                pass
            read_buffer.put(None)

        # Setup montage if needed
        if montage:
            left = w // 4
            w = w // 2
            lastframe = lastframe[:, left:left + w]

        # Padding for 32 multiple
        tmp = max(128, int(128 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)

        # Start threaded queues
        write_buffer = Queue(maxsize=500)
        read_buffer = Queue(maxsize=500)
        videogen = cv2_frame_generator(video_path)
        _thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
        _thread.start_new_thread(clear_write_buffer, (write_buffer,))

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

            # SSIM check
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

        if montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        else:
            write_buffer.put(lastframe)
        write_buffer.put(None)

        import time
        while not write_buffer.empty():
            time.sleep(0.1)

        pbar.close()
        vid_out.release()

        if transfer_audio and fps_not_assigned:
            try:
                self._transfer_audio(video_path, output_path)
            except Exception:
                # print("Audio transfer failed. Interpolated video will have no audio")

        return output_path
