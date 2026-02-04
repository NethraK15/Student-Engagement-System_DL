import torch
import torch.nn as nn
from torchvision import models

class FusionModel1(nn.Module):
    def __init__(self, openface_dim, num_classes=3):
        super(FusionModel1, self).__init__()

        # ----- CNN BRANCH (EfficientNet) -----
        self.cnn = models.efficientnet_b0(pretrained=True)

        # Remove classifier head
        self.cnn.classifier = nn.Identity()   # Output = 1280

        # ----- FUSION CLASSIFIER -----
        self.classifier = nn.Sequential(
            nn.Linear(1280 + openface_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, image, openface_feat):
        cnn_feat = self.cnn(image)            # [B, 1280]

        combined = torch.cat(
            (cnn_feat, openface_feat), dim=1
        )

        out = self.classifier(combined)
        return out
