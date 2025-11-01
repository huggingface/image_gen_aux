from PIL import Image
import glob
import re
# Natural sort (so 2 comes before 10)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

directory = "images/output"
# Collect all PNG files in the folder
frames = []
file_list = sorted(glob.glob(f"{directory}/img*.png"), key=natural_sort_key)

for filename in file_list:
    frames.append(Image.open(filename))

# Save as GIF
frames[0].save(
    "images/output/output.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,  # duration per frame in ms
    loop=0         # 0 = infinite loop
)
print(f"GIF saved as {directory}/output.gif")
