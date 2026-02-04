import torch
import torch.nn as nn
from torchvision import models

class FusionModel(nn.Module):
    def __init__(self, openface_dim, num_classes=3,
                 lstm_hidden=256, lstm_layers=1):
        super(FusionModel, self).__init__()

        # ----- CNN BRANCH -----
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()   # Output = 512

        # ----- LSTM -----
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # ----- FUSION CLASSIFIER -----
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden + openface_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, images, openface_feat):
        """
        images: [B, T, 3, H, W]
        openface_feat: [B, openface_dim]
        """

        B, T, C, H, W = images.shape

        # ---- CNN feature extraction per frame ----
        images = images.view(B * T, C, H, W)
        cnn_feat = self.cnn(images)          # [B*T, 512]
        cnn_feat = cnn_feat.view(B, T, 512)  # [B, T, 512]

        # ---- LSTM ----
        lstm_out, (h_n, c_n) = self.lstm(cnn_feat)

        # Take LAST time step output
        lstm_feat = lstm_out[:, -1, :]       # [B, lstm_hidden]

        # ---- Fusion ----
        combined = torch.cat(
            (lstm_feat, openface_feat), dim=1
        )

        out = self.classifier(combined)
        return out
