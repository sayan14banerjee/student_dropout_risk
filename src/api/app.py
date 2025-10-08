
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()
# Load model and columns
model = joblib.load(os.path.join("models", "best_model.pkl"))
columns = pd.read_csv(os.path.join("data", "processed", "X_test.csv")).columns.tolist()

class InputData(BaseModel):
	features: list

@app.post("/predict")
def predict(data: InputData):
	if len(data.features) != len(columns):
		raise HTTPException(status_code=400, detail=f"Expected {len(columns)} features, got {len(data.features)}")
	X = pd.DataFrame([data.features], columns=columns)
	pred = model.predict(X)
	return {"prediction": int(pred[0])}


