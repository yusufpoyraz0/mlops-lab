"""Bölüm 10: Modeli FastAPI ile servis et. Bölüm 13: istekleri logla."""
import json
import time
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
CLASSES = ["setosa", "versicolor", "virginica"]
LOG_FILE = Path("logs/predictions.jsonl")

model = mlflow.sklearn.load_model("model")
app = FastAPI(title="Iris MLOps Lab")


class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(f: Flower):
    row = pd.DataFrame(
        [[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]],
        columns=FEATURES,
    )
    idx = int(model.predict(row)[0])
    # Drift kontrolü için girdileri ve tahmini kaydet (klasör silinmişse yeniden oluştur)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open("a") as fh:
        fh.write(json.dumps({"ts": time.time(), **f.model_dump(), "prediction": idx}) + "\n")
    return {"class_id": idx, "class_name": CLASSES[idx]}