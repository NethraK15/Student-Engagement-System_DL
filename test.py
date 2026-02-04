import torch
from torch.utils.data import DataLoader
from dataset import EngagementDataset
from models.fusion_model import FusionModel

# 1️⃣ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2️⃣ Load dataset
dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat"
)

# Windows-safe DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

# 3️⃣ Get OpenFace feature dimension
_, openface_sample, _ = dataset[0]
openface_dim = openface_sample.shape[0]
print(f"OpenFace feature dim: {openface_dim}")

# 4️⃣ Load model
model = FusionModel(openface_dim=openface_dim).to(device)
model.load_state_dict(torch.load("models/engagement_fusion3.pth", map_location=device))
model.eval()
print("Model loaded and ready for evaluation.")

# 5️⃣ Accuracy calculation
correct = 0
total = 0

with torch.no_grad():  # very important
    for i, (images, openface_feats, labels) in enumerate(loader):
        print(f"Processing batch {i+1}/{len(loader)}")

        images = images.to(device)
        openface_feats = openface_feats.to(device)
        labels = labels.to(device)

        outputs = model(images, openface_feats)

        # Get predicted class
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
