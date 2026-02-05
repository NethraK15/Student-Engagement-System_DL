import torch
import torch.nn as nn
from torchvision import models

class FusionModel(nn.Module):
    def __init__(self, openface_dim, num_classes=3):
        super(FusionModel, self).__init__()

        # =======================
        # CNN IMAGE BRANCH (Frozen)
        # =======================
        # Use updated weights parameter to avoid deprecation warnings
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity()  # Output: 512-dim

        for param in self.cnn.parameters():
            param.requires_grad = False  # FREEZE CNN

        # =======================
        # OPENFACE BRANCH (Reduces high-dim features to 128)
        # =======================
        self.openface_net = nn.Sequential(
            nn.Linear(openface_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # =======================
        # FUSION CLASSIFIER
        # =======================
        # Now 512 (CNN) + 128 (processed OpenFace) = 640
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, openface_feat):
        # 1. Pass image through CNN -> [Batch, 512]
        cnn_feat = self.cnn(image)
        
        # 2. Pass OpenFace features through its branch -> [Batch, 128]
        # This fixes the "mat1 and mat2" shape mismatch!
        of_feat_reduced = self.openface_net(openface_feat)
        
        # 3. Concatenate both 
        # [Batch, 512 + 128] = [Batch, 640]
        combined = torch.cat((cnn_feat, of_feat_reduced), dim=1)
        
        # 4. Pass through final classifier
        out = self.classifier(combined)
        return out