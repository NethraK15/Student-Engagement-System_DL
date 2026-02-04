import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import EngagementDataset
from models.fusion_model_attention import AdaptiveAttentionFusion as FusionModel

# ===============================
# 1️⃣ Device setup
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">>> Using device:", device)

# ===============================
# 2️⃣ Load FULL dataset (for splitting)
# ===============================
full_dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat"
)

print(f">>> Full dataset size: {len(full_dataset)}")

# ===============================
# 3️⃣ Train / Test split (80 / 20)
# ===============================
indices = list(range(len(full_dataset)))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, shuffle=True
)

train_dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat",
    indices=train_idx
)

print(f">>> Training samples: {len(train_dataset)}")

# ===============================
# 4️⃣ DataLoader
# ===============================
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

# ===============================
# 5️⃣ Get OpenFace feature dimension
# ===============================
_, openface_sample, _ = train_dataset[0]
openface_dim = openface_sample.shape[0]
print(f">>> OpenFace feature dimension: {openface_dim}")

# ===============================
# 6️⃣ Initialize model
# ===============================
model = FusionModel(openface_dim=openface_dim).to(device)
print(">>> Model initialized")

# ===============================
# 7️⃣ Loss and optimizer
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ===============================
# 8️⃣ Training loop
# ===============================
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    print(f"\n>>> Epoch {epoch + 1}/{epochs} started")

    for batch_idx, (images, openface_feats, labels) in enumerate(train_loader):
        images = images.to(device)
        openface_feats = openface_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, openface_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f">>> Epoch {epoch + 1} completed | Avg Loss: {avg_loss:.4f}")

# ===============================
# 9️⃣ Save trained model
# ===============================
save_path = "models/engagement_fusion_attention.pth"
torch.save(model.state_dict(), save_path)
print(f"\n>>> Training complete")
print(f">>> Model saved to {save_path}")
