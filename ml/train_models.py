import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("ml/S_ALL_combined.csv")

print("Initial shape:", df.shape)

# =========================
# CLEAN LABELS
# =========================
df = df.dropna(subset=["Label"])
df["Label"] = df["Label"].astype(int)

# =========================
# KEEP ONLY NUMERIC FEATURES
# =========================
numeric_df = df.select_dtypes(include=[np.number])

print("Numeric shape:", numeric_df.shape)

# =========================
# CLEAN NaNs / Infs
# =========================
numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_df.dropna(inplace=True)

print("After cleaning:", numeric_df.shape)
print("Label distribution:")
print(numeric_df["Label"].value_counts())

# =========================
# SPLIT
# =========================
X = numeric_df.drop(columns=["Label"])
y = numeric_df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODELS
# =========================
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "SVM": SVC(
        kernel="rbf",
        probability=True
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

# =========================
# TRAIN & EVALUATE
# =========================
for name, model in models.items():
    print("\n" + "=" * 40)
    print(name)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
