import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from torchvision import transforms

class EngagementDataset(Dataset):
    def __init__(self, feature_csv, label_csv, images_dir):
        self.features = pd.read_csv(feature_csv)
        self.labels = pd.read_csv(label_csv)
        self.images_dir = images_dir

        image_files = sorted(os.listdir(images_dir))
        frame_to_filename = {i+1: image_files[i] for i in range(len(image_files))}
        self.features['filename'] = self.features['frame'].map(frame_to_filename)
        
        self.data = self.features.merge(self.labels, on="filename", how="inner")

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        metadata_cols = ["filename", "label", "frame", " face_id", " timestamp", " confidence", " success"]
        self.feature_cols = [col for col in self.data.columns if col not in metadata_cols]

    def apply_illumination_normalization(self, img):
        """
        Applies CLAHE and a basic Retinex-based normalization to 
        reduce sensitivity to classroom lighting.
        """
        # 1. Convert to LAB color space to process Luminance (L)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # 2. Histogram Equalization (CLAHE)
        # Prevents over-amplification of noise in dark areas
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # 3. Simple Retinex-like Normalization (Log transform)
        # Helps in recovering details in shadows/highlights
        img_log = np.log1p(cl.astype(np.float32))
        cl = cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Merge back
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # 4. Optional: Gradient-based Edge Enhancement (Sharpening)
        # Highlights structural features (eyes, mouth) regardless of light
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        final_img = cv2.filter2D(final_img, -1, kernel)

        return final_img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.images_dir, row["filename"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Novelty Extension 3
        image = self.apply_illumination_normalization(image)

        image = self.transform(image)

        openface_feat = row[self.feature_cols].values.astype(float)
        openface_feat = torch.tensor(openface_feat, dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)

        return image, openface_feat, label