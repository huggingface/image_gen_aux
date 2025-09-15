import cv2
import torch
from torch.nn import functional as F
from model.RIFE_HDv2 import Model  # Use the specific model version you need

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = Model()
model.load_model('train_log', -1)  # Replace 'train_log' with your model directory
model.eval()
model.device()

# Load and preprocess images
img0 = cv2.imread('input0.png', cv2.IMREAD_UNCHANGED)  # Replace with your input image paths
img1 = cv2.imread('input1.png', cv2.IMREAD_UNCHANGED)
img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

# Pad images to be divisible by 32
n, c, h, w = img0.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
img0 = F.pad(img0, padding)
img1 = F.pad(img1, padding)

# Perform inference
middle = model.inference(img0, img1)

# Save the output
output = (middle[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
cv2.imwrite('output.png', output)  # Replace with your desired output path