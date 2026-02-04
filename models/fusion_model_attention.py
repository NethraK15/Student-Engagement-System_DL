import torch
import torch.nn as nn
import timm


class AdaptiveAttentionFusion(nn.Module):
    def __init__(self, openface_dim, num_classes=3):
        super(AdaptiveAttentionFusion, self).__init__()

        # ===============================
        # 1️⃣ CNN Backbone (EfficientNet-B0)
        # ===============================
        self.cnn = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0  # remove classifier head
        )
        cnn_feature_dim = self.cnn.num_features

        # ===============================
        # 2️⃣ OpenFace Feature Encoder
        # ===============================
        self.openface_fc = nn.Sequential(
            nn.Linear(openface_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ===============================
        # 3️⃣ Adaptive Attention Module (NOVEL)
        # α ∈ [0,1]
        # ===============================
        self.attention = nn.Sequential(
            nn.Linear(cnn_feature_dim + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # ===============================
        # 4️⃣ Fusion Classifier
        # ===============================
        self.classifier = nn.Sequential(
            nn.Linear(cnn_feature_dim + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, openface_feats):
        # -------------------------------
        # Extract features
        # -------------------------------
        img_features = self.cnn(images)                 # F_img
        face_features = self.openface_fc(openface_feats)  # F_face

        # -------------------------------
        # Compute attention weight α
        # -------------------------------
        concat = torch.cat([img_features, face_features], dim=1)
        alpha = self.attention(concat)

        # -------------------------------
        # Adaptive fusion
        # -------------------------------
        fused_features = torch.cat(
            [alpha * img_features, (1 - alpha) * face_features],
            dim=1
        )

        # -------------------------------
        # Classification
        # -------------------------------
        output = self.classifier(fused_features)
        return output
