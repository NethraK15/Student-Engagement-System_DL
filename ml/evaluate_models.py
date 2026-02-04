import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
df = pd.read_csv("ml/S_ALL_combined.csv")

# -----------------------------
# 2️⃣ Handle missing values
# -----------------------------

# Drop rows where the label is missing
df = df.dropna(subset=["Label"])

# Separate features and label
X = df.drop(columns=["Label"])
y = df["Label"]

# Keep only numeric features
X = X.select_dtypes(include=["number"])

# Fill missing numeric values with mean
X = X.fillna(X.mean())

# -----------------------------
# 3️⃣ Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4️⃣ Scale features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5️⃣ Load saved models
# -----------------------------
models = {
    "Random Forest": joblib.load("ml/models/random_forest.pkl"),
    "SVM": joblib.load("ml/models/svm.pkl"),
    "Gradient Boosting": joblib.load("ml/models/gradient_boosting.pkl")
}

# -----------------------------
# 6️⃣ Evaluate models
# -----------------------------
for name, model in models.items():
    preds = model.predict(X_test)

    print("\n" + "="*40)
    print(f"Model: {name}")
    print("="*40)

    print(f"Accuracy  : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision : {precision_score(y_test, preds, average='weighted'):.4f}")
    print(f"Recall    : {recall_score(y_test, preds, average='weighted'):.4f}")
    print(f"F1-score  : {f1_score(y_test, preds, average='weighted'):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))
