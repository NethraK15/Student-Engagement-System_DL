import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import EngagementDataset

def visualize_extension(image_path):
    # 1. Load Original
    raw_img = cv2.imread(image_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    # 2. Simulate a "Dark Classroom" (Low Light)
    dark_img = (raw_img.astype(np.float32) * 0.3).astype(np.uint8)
    
    # 3. Simulate "Glare/Overexposure" (Bright Light)
    bright_img = cv2.add(raw_img, np.array([70.0]))

    # 4. Use your Dataset's logic to fix them
    # (Assuming you updated dataset.py as shown in previous message)
    ds = EngagementDataset("dataset/openface/images_flat.csv", "dataset/labels.csv", "dataset/images_flat")
    
    fixed_dark = ds.apply_illumination_normalization(dark_img)
    fixed_bright = ds.apply_illumination_normalization(bright_img)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Dark Lighting
    axes[0, 0].imshow(raw_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(dark_img)
    axes[0, 1].set_title("Simulated Dark Classroom")
    axes[0, 2].imshow(fixed_dark)
    axes[0, 2].set_title("Extension 3: Recovered Features")

    # Row 2: Bright Lighting
    axes[1, 0].imshow(raw_img)
    axes[1, 0].set_title("Original Image")
    axes[1, 1].imshow(bright_img)
    axes[1, 1].set_title("Simulated Window Glare")
    axes[1, 2].imshow(fixed_bright)
    axes[1, 2].set_title("Extension 3: Normalized Contrast")

    for ax in axes.ravel(): ax.axis('off')
    plt.tight_layout()
    plt.show()

# Run it on one of your frames
visualize_extension("dataset/images_flat/img1_113.jpg")