import torch
print(torch.__version__)          # should show the version you installed
print(torch.version.cuda)         # should say 12.1
print(torch.cuda.is_available())  # should be True
print(torch.cuda.get_device_name(0))  # should show "NVIDIA GeForce RTX 4050 GPU"

from image_gen_aux import BEN2BackgroundRemover
from image_gen_aux.utils import load_image


model = BEN2BackgroundRemover.from_pretrained("PramaLLC/BEN2").to("cuda")

image = load_image(
    "https://archive.smashing.media/assets/344dbf88-fdf9-42bb-adb4-46f01eedd629/68dd54ca-60cf-4ef7-898b-26d7cbe48ec7/10-dithering-opt.jpg"
)

foreground = model(image)[0]
foreground.save("foreground.png")