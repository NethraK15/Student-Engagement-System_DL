import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import EngagementDataset
from models.fusion_model_eff import FusionModel

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Using device: {device}")

# 2. Load Full Dataset
full_dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat"
)

# 3. Create Train/Test Split (80% Train, 20% Test)
indices = list(range(len(full_dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

print(f">>> Total: {len(full_dataset)} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# 4. Get OpenFace feature dimension
_, openface_sample, _ = full_dataset[0]
openface_dim = openface_sample.shape[0]

# 5. Model Initialization
model = FusionModel(openface_dim=openface_dim).to(device)

# Freeze CNN (to focus on the fusion and classifier)
for param in model.cnn.parameters():
    param.requires_grad = False

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# 6. Training Loop
epochs = 10 # Increased slightly since we are splitting data
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, openface_feats, labels in train_loader:
        images, openface_feats, labels = images.to(device), openface_feats.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, openface_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 7. Validation (Test Accuracy) after each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, openface_feats, labels in test_loader:
            images, openface_feats, labels = images.to(device), openface_feats.to(device), labels.to(device)
            outputs = model(images, openface_feats)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {val_acc:.2f}%")

# 8. Save
torch.save(model.state_dict(), "models/engagement_fusion_novelty.pth")
print(">>> Training complete. Model saved.")