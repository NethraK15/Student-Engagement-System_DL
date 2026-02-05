import torch
import torch.nn as nn
from torchvision import models

class FusionModel(nn.Module):
    def __init__(self, openface_dim, num_classes=3):
        super(FusionModel, self).__init__()

        # =======================
        # CNN IMAGE BRANCH: EfficientNet-B0
        # =======================
        # Using B0 because it is highly efficient for student engagement tasks
        self.cnn = models.efficientnet_b0(pretrained=True)
        
        # EfficientNet stores its output size in .classifier[1].in_features
        # For B0, this is 1280. We remove the classification head:
        self.cnn_out_dim = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity() 

        # FREEZE CNN (Keep pre-trained ImageNet features)
        for param in self.cnn.parameters():
            param.requires_grad = False

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
        # Combined dim = 1280 (EfficientNet) + 128 (OpenFace) = 1408
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, openface_feat):
        # 1. Image features -> [Batch, 1280]
        cnn_feat = self.cnn(image)
        
        # 2. OpenFace features -> [Batch, 128]
        of_feat_reduced = self.openface_net(openface_feat)
        
        # 3. Concatenate
        combined = torch.cat((cnn_feat, of_feat_reduced), dim=1)
        
        # 4. Final Classification
        out = self.classifier(combined)
        return out