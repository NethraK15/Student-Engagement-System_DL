import torch
from torch.utils.data import DataLoader
from dataset import EngagementDataset
from models.fusion_model import FusionModel

# ---------- FORCE CPU (IMPORTANT FOR DEBUG) ----------
device = torch.device("cpu")
print(">>> Using device:", device)

# ---------- LOAD DATASET ----------
print(">>> Loading dataset...")
dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat"
)
print(f">>> Dataset loaded with {len(dataset)} samples")

# ---------- DATALOADER (SAFE SETTINGS) ----------
loader = DataLoader(
    dataset,
    batch_size=4,        # VERY IMPORTANT
    shuffle=True,
    num_workers=0,       # WINDOWS SAFE
    pin_memory=False
)
print(">>> DataLoader created")

# ---------- OPENFACE DIM ----------
_, openface_sample, _ = dataset[0]
openface_dim = openface_sample.shape[0]
print(">>> OpenFace feature dimension:", openface_dim)

# ---------- MODEL ----------
print(">>> Initializing model...")
model = FusionModel(openface_dim=openface_dim).to(device)

# FREEZE CNN (CRITICAL)
for param in model.cnn.parameters():
    param.requires_grad = False
print(">>> CNN frozen")

# ---------- LOSS + OPTIMIZER ----------
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

epochs = 10

# ---------- TRAIN ----------
for epoch in range(epochs):
    model.train()
    total_loss = 0

    print(f">>> Epoch {epoch+1} started")

    for batch_idx, (images, openface_feats, labels) in enumerate(loader):
        images = images.to(device)
        openface_feats = openface_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, openface_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"    Batch {batch_idx} | Loss {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(loader):.4f}")

# ---------- SAVE ----------
torch.save(model.state_dict(), "models/engagement_fusion3.pth")
print(">>> Training complete")
