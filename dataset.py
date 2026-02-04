import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms


class EngagementDataset(Dataset):
    def __init__(self, feature_csv, label_csv, images_dir, indices=None):

        # ===============================
        # 1️⃣ Load CSV files
        # ===============================
        self.features = pd.read_csv(feature_csv)
        self.labels = pd.read_csv(label_csv)
        self.images_dir = images_dir

        # ===============================
        # 2️⃣ Map frame number → filename
        # ===============================
        image_files = sorted(os.listdir(images_dir))
        frame_to_filename = {i + 1: image_files[i] for i in range(len(image_files))}
        self.features["filename"] = self.features["frame"].map(frame_to_filename)

        # ===============================
        # 3️⃣ Merge features + labels
        # ===============================
        self.data = self.features.merge(
            self.labels, on="filename", how="inner"
        )

        # ===============================
        # 4️⃣ APPLY TRAIN / TEST SPLIT (KEY FIX)
        # ===============================
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)

        # ===============================
        # 5️⃣ Image preprocessing
        # ===============================
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ===============================
        # 6️⃣ OpenFace feature columns
        # ===============================
        metadata_cols = [
            "filename", "label", "frame",
            " face_id", " timestamp", " confidence", " success"
        ]

        self.feature_cols = [
            col for col in self.data.columns
            if col not in metadata_cols
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ===============================
        # IMAGE
        # ===============================
        img_path = os.path.join(self.images_dir, row["filename"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        # ===============================
        # OpenFace FEATURES
        # ===============================
        openface_feat = row[self.feature_cols].values.astype(float)
        openface_feat = torch.tensor(openface_feat, dtype=torch.float32)

        # ===============================
        # LABEL
        # ===============================
        label = torch.tensor(row["label"], dtype=torch.long)

        return image, openface_feat, label
