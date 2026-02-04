import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class FusionModel(nn.Module):
    def __init__(self, openface_dim, num_classes=3, lstm_hidden=256):
        super().__init__()

        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()   # 512

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden + openface_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, openface_feats):
        """
        images: [B, T, 3, H, W] or [B, 3, H, W]
        """
        # Support both single-frame [B, C, H, W] and sequence [B, T, C, H, W]
        if images.dim() == 4:
            images = images.unsqueeze(1)  # [B, 1, C, H, W]
        
        B, T, C, H, W = images.shape

        images = images.view(B * T, C, H, W)
        cnn_feat = self.cnn(images)
        cnn_feat = cnn_feat.view(B, T, 512)

        lstm_out, _ = self.lstm(cnn_feat)
        lstm_feat = lstm_out[:, -1, :]

        combined = torch.cat([lstm_feat, openface_feats], dim=1)
        return self.classifier(combined)
