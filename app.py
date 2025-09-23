# import torch
# print(torch.__version__)          # should show the version you installed
# print(torch.version.cuda)         # should say 12.1
# print(torch.cuda.is_available())  # should be True
# print(torch.cuda.get_device_name(0))  # should show "NVIDIA GeForce RTX 4050 GPU"

# from image_gen_aux import BEN2BackgroundRemover
# from image_gen_aux.utils import load_image

# # Load the BEN2 model from your local weights
# model = BEN2BackgroundRemover.from_pretrained(r"C:\1himan\Projects\HuggingFace\RIFE_implementation\image_gen_aux").to("cuda")

# # Load an image for background removal
# image = load_image("https://images.pexels.com/photos/236599/pexels-photo-236599.jpeg?cs=srgb&dl=pexels-pixabay-236599.jpg&fm=jpg")  # Replace with your actual image path

# # Remove background (returns foreground with transparent background)
# foreground = model(image)[0]

# # Save the result
# foreground.save("foreground_no_background.png")

# print("Background removal complete! Check 'foreground_no_background.png'")

# ----------------------------------------------------------------------------------------

from image_gen_aux import RIFE
# # can you find the links to the weight of this model?
model = RIFE.from_pretrained(r"C:\1himan\Projects\HuggingFace\RIFE_implementation\image_gen_aux").to("cuda")

# print("Model loaded.", model)
# # interpolated = model.interpolate_images(img0, img1)[0]
# # interpolated.save("interpolated.png")