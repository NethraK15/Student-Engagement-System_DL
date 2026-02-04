import os
import pandas as pd
from feature_extractor import EngagementFeatureExtractor

# This gets the folder where main_run.py is actually located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define relative paths based on your project structure
# This will work regardless of whether your project is on C:, D:, or a USB drive
BASE_PATH = os.path.join(BASE_DIR, "dataset", "images")
OUTPUT_PATH = os.path.join(BASE_DIR, "output")

# Mapping your folder names to integer labels
FOLDERS = {
    "0_disengaged": 0,
    "1_partial": 1,
    "2_engaged": 2
}

def main():
    print("BASE_PATH:", BASE_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    extractor = EngagementFeatureExtractor()
    master_data = []

    for folder_name, label_id in FOLDERS.items():
        folder_full_path = os.path.join(BASE_PATH, folder_name)
        print(f"\nChecking folder: {folder_full_path}")
        print("Exists:", os.path.exists(folder_full_path))

        if os.path.exists(folder_full_path):
            print("Files:", os.listdir(folder_full_path))

        folder_results = extractor.extract_from_folder(folder_full_path, label_id)
        print("Extracted rows:", len(folder_results))

        master_data.extend(folder_results)


    # Creating the final DataFrame
    if master_data:
        df = pd.DataFrame(master_data)
        
        # Save individual version
        csv_file = os.path.join(OUTPUT_PATH, "mediapipe_features.csv")
        df.to_csv(csv_file, index=False)
        
        print("-" * 30)
        print(f"SUCCESS!")
        print(f"Total images processed: {len(df)}")
        print(f"Features saved to: {csv_file}")
    else:
        print("No landmarks were detected. Check your image paths or quality.")

    extractor.close()

if __name__ == "__main__":
    main()