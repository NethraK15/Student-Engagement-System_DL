import cv2
import mediapipe as mp
import os
import pandas as pd
import mediapipe as mp
print(mp.__file__)
exit()

class EngagementFeatureExtractor:
    def __init__(self):
        # Initializing the FaceMesh solution
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract_from_folder(self, folder_path, label_value):
        data_list = []
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found -> {folder_path}")
            return []

        print(f"Processing Label {label_value}: {os.path.basename(folder_path)}...")

        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, file)
                image = cv2.imread(img_path)
                if image is None: continue

                # MediaPipe requires RGB images
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(img_rgb)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    # Start the row with metadata
                    row = {"ImageID": file, "Label": label_value}
                    
                    ih, iw, _ = image.shape
                    for i, lm in enumerate(landmarks):
                        # Converting normalized coordinates to pixel coordinates
                        row[f"x{i}"] = int(lm.x * iw)
                        row[f"y{i}"] = int(lm.y * ih)
                        row[f"z{i}"] = lm.z # Depth relative to head center
                    
                    data_list.append(row)
        return data_list

    def close(self):
        self.face_mesh.close()