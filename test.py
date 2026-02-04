import torch
from torch.utils.data import DataLoader
from dataset import EngagementDataset
from models.fusion_model import FusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = EngagementDataset(
    feature_csv="dataset/openface/images_flat.csv",
    label_csv="dataset/labels.csv",
    images_dir="dataset/images_flat",
    seq_len=16
)

loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

_, openface_sample, _ = dataset[0]
openface_dim = openface_sample.shape[0]

model = FusionModel(openface_dim).to(device)
model.load_state_dict(torch.load("models/engagement_lstm_final.pth", map_location=device))
model.eval()

correct = total = 0

with torch.no_grad():
    for images, openface, labels in loader:
        images, openface, labels = images.to(device), openface.to(device), labels.to(device)
        outputs = model(images, openface)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {100 * correct / total:.2f}%")
