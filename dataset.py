import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms
import numpy as np

class EngagementDataset(Dataset):
    def __init__(self, feature_csv, label_csv, images_dir, seq_len=16):
        self.seq_len = seq_len
        self.images_dir = images_dir

        features = pd.read_csv(feature_csv)
        labels = pd.read_csv(label_csv)

        # Map frame → filename
        image_files = sorted(os.listdir(images_dir))
        frame_to_filename = {i+1: image_files[i] for i in range(len(image_files))}
        features["filename"] = features["frame"].map(frame_to_filename)

        data = features.merge(labels, on="filename", how="inner")
        data = data.sort_values("frame").reset_index(drop=True)

        self.data = data

        # Columns
        metadata_cols = ["filename", "label", "frame", " face_id", " timestamp", " confidence", " success"]
        self.feature_cols = [c for c in data.columns if c not in metadata_cols]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        seq_data = self.data.iloc[idx : idx + self.seq_len]

        images = []
        openface_feats = []

        for _, row in seq_data.iterrows():
            img_path = os.path.join(self.images_dir, row["filename"])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            images.append(img)

            feat = row[self.feature_cols].values.astype(float)
            openface_feats.append(feat)

        images = torch.stack(images)                    # [T, 3, H, W]
        openface_feats = torch.tensor(
            np.mean(openface_feats, axis=0), dtype=torch.float32
        )

        label = torch.tensor(seq_data.iloc[-1]["label"], dtype=torch.long)

        return images, openface_feats, label
