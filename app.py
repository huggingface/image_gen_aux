import torch
print(torch.__version__)          # should show the version you installed
print(torch.version.cuda)         # should say 12.1
print(torch.cuda.is_available())  # should be True
print(torch.cuda.get_device_name(0))  # should show "NVIDIA GeForce RTX 4050 GPU"

# from image_gen_aux import BEN2BackgroundRemover
# from image_gen_aux.utils import load_image

# # Load the BEN2 model from your local weights
# model = BEN2BackgroundRemover.from_pretrained(r"C:\1himan\Projects\HuggingFace\RIFE_implementation\image_gen_aux").to("cuda")

# # Load an image for background removal
# image = load_image("https://images.pexels.com/photos/236599/pexels-photo-236599.jpeg?cs=srgb&dl=pexels-pixabay-236599.jpg&fm=jpg")  # Replace with your actual image path

# # Remove background (returns foreground with transparent background)
# # here the "loaded model" is used as a callable object
# foreground = model(image)[0]

# # Save the result
# foreground.save("foreground_no_background.png")

# print("Background removal complete! Check 'foreground_no_background.png'")

# ----------------------------------------------------------------------------------------

from image_gen_aux import RIFE
# using from_pretrained
model=RIFE.from_pretrained("1himan/RIFE")

# using load_model:
# model=RIFE("./")

# For image interpolation:
outputs = model.interpolate_image("images/img0.png", "images/img1.png", exp=4)
print(f"Generated {len(outputs)} interpolated images")

# For video interpolation:
output_video = model.interpolate_video(
    video_path="videos/spider-man-electro.gif",
    output_path="videos/output_video.mp4",
    exp=2,
    scale=1.0,
    transfer_audio=True,
    # montage=True
)
print(f"Video saved to: {output_video}")