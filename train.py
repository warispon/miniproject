# train.py
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_linear_pipeline.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# ต้องมีคอลัมน์อย่างน้อยตามนี้ (Kaggle Titanic train.csv มีแน่นอน)
needed_cols = ["Survived", "Pclass", "Sex", "Age", "Fare", "SibSp"]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"CSV ของปาล์มขาดคอลัมน์: {missing}\n"
        f"คอลัมน์ที่ต้องมี: {needed_cols}"
    )

# ใช้แค่ 5 features + target
df = df[needed_cols].copy()

# -----------------------------
# Clean & Encode Sex -> 0/1
# -----------------------------
df["Sex"] = df["Sex"].astype(str).str.strip().str.lower()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# บังคับให้เป็นตัวเลข (กันพวกค่าแปลก/เว้นวรรค/ตัวอักษร)
for col in ["Survived", "Pclass", "Sex", "Age", "Fare", "SibSp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ลบแถวที่ target หรือฟีเจอร์หลักหาย (Sex ต้องไม่หาย ไม่งั้นโมเดลใช้ไม่ได้)
df = df.dropna(subset=["Survived", "Pclass", "Sex"])

# -----------------------------
# Split X, y
# -----------------------------
X = df[["Pclass", "Sex", "Age", "Fare", "SibSp"]]
y = df["Survived"].astype(int)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Preprocess (median impute + scale) for numeric columns
# -----------------------------
num_features = ["Pclass", "Sex", "Age", "Fare", "SibSp"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
    ],
    remainder="drop"
)

# -----------------------------
# Model: Multiple Linear Regression
# -----------------------------
model = LinearRegression()

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

# -----------------------------
# Train
# -----------------------------
pipe.fit(X_train, y_train)

# -----------------------------
# Predict: regression output -> clip 0..1 -> threshold 0.5
# -----------------------------
y_pred_cont = pipe.predict(X_test)
y_proba = np.clip(y_pred_cont, 0, 1)
y_pred = (y_proba >= 0.5).astype(int)

# -----------------------------
# Evaluate
# -----------------------------
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

mae = mean_absolute_error(y_test, y_proba)
mse = mean_squared_error(y_test, y_proba)

print("=== Titanic Survival (Multiple Linear Regression) ===")
print("Features: Pclass, Sex(0/1), Age, Fare, SibSp")
print(f"Rows used (after cleaning): {len(df)}")
print()
print(f"Accuracy (threshold 0.5): {acc:.4f}")
print("Confusion Matrix:")
print(cm)
print(f"ROC-AUC (using clipped output): {auc:.4f}")
print(f"MAE (probability): {mae:.4f}")
print(f"MSE (probability): {mse:.4f}")

# -----------------------------
# Save model
# -----------------------------
joblib.dump(pipe, MODEL_PATH)
print(f"\nModel saved -> {MODEL_PATH}")
