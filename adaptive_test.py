import torch
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
# 2️⃣ Load FULL dataset (for same split)
# ===============================
full_dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat"
)

print(f">>> Full dataset size: {len(full_dataset)}")

# ===============================
# 3️⃣ Same train / test split (IMPORTANT)
# ===============================
indices = list(range(len(full_dataset)))
_, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, shuffle=True
)

test_dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat",
    indices=test_idx
)

print(f">>> Test samples: {len(test_dataset)}")

# ===============================
# 4️⃣ DataLoader
# ===============================
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# ===============================
# 5️⃣ Get OpenFace feature dimension
# ===============================
_, openface_sample, _ = test_dataset[0]
openface_dim = openface_sample.shape[0]
print(f">>> OpenFace feature dimension: {openface_dim}")

# ===============================
# 6️⃣ Load trained model
# ===============================
model = FusionModel(openface_dim=openface_dim).to(device)

model_path = "models/engagement_fusion_attention.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f">>> Model loaded from {model_path}")

# ===============================
# 7️⃣ Evaluation
# ===============================
correct = 0
total = 0

with torch.no_grad():
    for images, openface_feats, labels in test_loader:
        images = images.to(device)
        openface_feats = openface_feats.to(device)
        labels = labels.to(device)

        outputs = model(images, openface_feats)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n>>> REAL Test Accuracy (Attention Model): {accuracy:.2f}%")
