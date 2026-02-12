import os
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd

app = FastAPI()

# CORS เผื่อเปิด index.html แบบไฟล์แล้วเรียก API ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load model (robust path)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> miniproject/
MODEL_PATH = os.path.join(BASE_DIR, "model", "titanic_linear_pipeline.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        f"ให้รัน: python train.py ก่อน เพื่อสร้างไฟล์โมเดล"
    )

pipe = joblib.load(MODEL_PATH)

# -----------------------------
# Serve web page
# -----------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))  # -> miniproject/backend/
INDEX_PATH = os.path.join(BACKEND_DIR, "index.html")

@app.get("/")
def home():
    return FileResponse(INDEX_PATH)

# -----------------------------
# API
# -----------------------------
class Input(BaseModel):
    pclass: int      # 1/2/3
    sex: int         # 0=male, 1=female
    age: float
    fare: float
    sibsp: int

@app.post("/predict")
def predict(inp: Input):
    X = pd.DataFrame([{
        "Pclass": inp.pclass,
        "Sex": inp.sex,
        "Age": inp.age,
        "Fare": inp.fare,
        "SibSp": inp.sibsp
    }])

    pred = float(pipe.predict(X)[0])
    proba = float(np.clip(pred, 0, 1))
    label = int(proba >= 0.5)

    return {
        "survival_probability": round(proba, 4),
        "prediction_label": label,
        "prediction_text": "Survived" if label == 1 else "Not Survived"
    }
