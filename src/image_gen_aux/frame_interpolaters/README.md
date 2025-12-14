# RIFE interpolaters

Usage

```py
from image_gen_aux import RIFE
# using from_pretrained
model=RIFE.from_pretrained("1himan/RIFE")

# using load_model:
# model=RIFE("./")

# For image interpolation:
# provide the correct video path, according to how you compiled this project
outputs = model.interpolate_image("images/img0.png", "images/img1.png", exp=4)
print(f"Generated {len(outputs)} interpolated images")

# For video interpolation:
output_video = model.interpolate_video(
    # provide the correct video path, according to how you compiled this project
    video_path="videos/spider-man-electro.gif",
    output_path="videos/output_video.mp4",
    exp=2,
    scale=1.0,
    transfer_audio=True,
    # montage=True
)
print(f"Video saved to: {output_video}")
```