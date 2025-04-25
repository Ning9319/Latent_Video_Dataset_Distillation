import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Path to the .pt file
pt_file_path = "./synthetic_data.pt"

# Load the tensor
image_tensor = torch.load(pt_file_path, weights_only=True).to(torch.float32)

# Verify the tensor shape
print(f"Tensor shape: {image_tensor.shape}")  # Expected: [51, 16, 3, 112, 112]


# 
class_i = 2
video_tensor = image_tensor[class_i] # First class' video
print("video_tensor has type", video_tensor.dtype)


# Convert to numpy and transpose for visualization (frame, height, width, channels)
video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()

# Normalize to [0, 1] if necessary
video_np = (video_np - video_np.min()) / (video_np.max() - video_np.min())

# Create a figure for the animation
fig, ax = plt.subplots()
frame_image = ax.imshow(video_np[0])  # Show the first frame initially
ax.axis("off")  # Turn off axes for better visualization

# Function to update each frame in the animation
def update_frame(frame_idx):
    frame_image.set_data(video_np[frame_idx])
    return [frame_image]

# Create the animation
ani = animation.FuncAnimation(
    fig, update_frame, frames=video_np.shape[0], interval=200, blit=True
)

plt.show()
