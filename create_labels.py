import os
import csv

base_dir = "dataset/images"
label_map = {
    "0_disengaged": 0,
    "1_partial": 1,
    "2_engaged": 2
}

rows = []

for folder, label in label_map.items():
    folder_path = os.path.join(base_dir, folder)
    for img in os.listdir(folder_path):
        rows.append([img, label])

with open("dataset/labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)

print("labels.csv created")
