import torch
from torch.utils.data import DataLoader, random_split
from dataset import EngagementDataset
from models.fusion_model import FusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- DATASET ----------
dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat",
    seq_len=16
)

print("Dataset size:", len(dataset))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)

# ---------- OPENFACE DIM ----------
_, openface_sample, _ = dataset[0]
openface_dim = openface_sample.shape[0]

# ---------- MODEL ----------
model = FusionModel(openface_dim).to(device)

# Freeze CNN
for p in model.cnn.parameters():
    p.requires_grad = False

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

epochs = 1

# ---------- TRAIN ----------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch+1} started")

    for batch_idx, (images, openface, labels) in enumerate(train_loader):
        print(f"  Processing batch {batch_idx}")

        images = images.to(device)
        openface = openface.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, openface)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "models/engagement_lstm_final.pth")
print("Training complete.")
