import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import EngagementDataset
from models.fusion_model_eff import FusionModel

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load dataset
full_dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat"
)

# 3. Use the EXACT SAME SPLIT logic as train.py
indices = list(range(len(full_dataset)))
_, test_idx = train_test_split(indices, test_size=0.2, random_state=42) # random_state is key!
test_dataset = Subset(full_dataset, test_idx)

loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

# 4. Get OpenFace feature dimension
_, openface_sample, _ = full_dataset[0]
openface_dim = openface_sample.shape[0]

# 5. Load model
model = FusionModel(openface_dim=openface_dim).to(device)
# Make sure to load the 'novelty' version you just trained
model.load_state_dict(torch.load("models/engagement_fusion_novelty.pth", map_location=device))
model.eval()
print(f"Model loaded. Testing on {len(test_dataset)} unseen images.")

# 6. Evaluation Loop
correct = 0
total = 0

with torch.no_grad():
    for images, openface_feats, labels in loader:
        images = images.to(device)
        openface_feats = openface_feats.to(device)
        labels = labels.to(device)

        outputs = model(images, openface_feats)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Final Test Accuracy: {accuracy:.2f}%")