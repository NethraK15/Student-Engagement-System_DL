import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def calculate_ear(landmarks, eye_indices):
    # Replicates Eye-based Behavioral Features
    pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        return None
    
    lm = results.multi_face_landmarks[0].landmark
    
    # Calculate Behavioral metrics to substitute OpenFace AUs
    left_ear = calculate_ear(lm, [362, 385, 387, 263, 373, 380])
    right_ear = calculate_ear(lm, [33, 160, 158, 133, 153, 144])
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Create feature row
    row = {"ImageID": os.path.basename(image_path), "EAR": avg_ear}
    
    # Add the 468 landmarks (The MediaPipe part of the paper)
    for i in range(468):
        row[f"x{i}"] = lm[i].x
        row[f"y{i}"] = lm[i].y
        
    return row

# Main loop to process your folders
dataset_path = r"dataset/images"
output_data = []

for label in ["0_disengaged", "1_partial", "2_engaged"]:
    folder = os.path.join(dataset_path, label)
    label_id = int(label[0])
    print(f"Processing {label}...")
    
    for img_name in os.listdir(folder):
        res = extract_features(os.path.join(folder, img_name))
        if res:
            res["Label"] = label_id
            output_data.append(res)

df = pd.DataFrame(output_data)
df.to_csv("replicated_features.csv", index=False)
print("Finished! You now have the Behavioral + Landmark features needed for the paper.")