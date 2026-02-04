import os
import glob
import pandas as pd

# =========================
# PROJECT ROOT
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

print("PROJECT ROOT:", PROJECT_ROOT)

# =========================
# CORRECT PATHS
# =========================
mediapipe_path = os.path.join(
    PROJECT_ROOT, "output", "mediapipe_features.csv"
)

openface_dir = os.path.join(
    PROJECT_ROOT, "dataset", "openface"
)

ml_dir = os.path.join(PROJECT_ROOT, "ml")

print("MediaPipe CSV:", mediapipe_path)
print("OpenFace DIR :", openface_dir)

# =========================
# VALIDATION
# =========================
assert os.path.exists(mediapipe_path), "❌ mediapipe_features.csv NOT FOUND"
assert os.path.exists(openface_dir), "❌ openface folder NOT FOUND"

# =========================
# LOAD MEDIAPIPE
# =========================
mediapipe_df = pd.read_csv(mediapipe_path)

assert "Label" in mediapipe_df.columns, "❌ 'label' column missing"

# =========================
# LOAD OPENFACE CSV (AUTO)
# =========================
openface_csvs = glob.glob(os.path.join(openface_dir, "*.csv"))
print("OpenFace CSVs found:", openface_csvs)

assert len(openface_csvs) > 0, "❌ No OpenFace CSV found"

openface_df = pd.read_csv(openface_csvs[0])

# =========================
# FEATURE SETS (DIAGRAM MATCH)
# =========================

# S1 – Facial landmarks (MediaPipe)
S1 = mediapipe_df.drop(columns=["Label"])

# S2 – Eye gaze + Head pose (OpenFace)
S2 = openface_df[
    [c for c in openface_df.columns if "gaze" in c or c.startswith("pose_")]
]

# S3 – Action Units (OpenFace)
S3 = openface_df[
    [c for c in openface_df.columns if c.startswith("AU")]
]

labels = mediapipe_df["Label"]

# Combined
S_ALL = pd.concat([S1, S2, S3], axis=1)

# =========================
# ADD LABEL
# =========================
S1["Label"] = labels
S2["Label"] = labels
S3["Label"] = labels
S_ALL["Label"] = labels

# =========================
# SAVE
# =========================
os.makedirs(ml_dir, exist_ok=True)

S1.to_csv(os.path.join(ml_dir, "S1_facelandmarks.csv"), index=False)
S2.to_csv(os.path.join(ml_dir, "S2_gaze_headpose.csv"), index=False)
S3.to_csv(os.path.join(ml_dir, "S3_actionunits.csv"), index=False)
S_ALL.to_csv(os.path.join(ml_dir, "S_ALL_combined.csv"), index=False)

print("✅ Feature extraction completed successfully")
