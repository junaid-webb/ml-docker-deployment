import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris ML API")

bundle = joblib.load("model.joblib")
model = bundle["model"]
classes = bundle["classes"]


class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "ML model API is running", "docs": "/docs", "version": "1.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(f: Features):
    row = [[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]]
    idx = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    return {
        "prediction": classes[idx],
        "confidence": round(float(proba[idx]), 3),
    }
