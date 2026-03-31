import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Define the input schema using Pydantic
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 2. Initialize the FastAPI app
app = FastAPI(title="Iris Species Predictor")

# 3. Load the model and metadata at startup
try:
    model_data = joblib.load("iris_model.joblib")
    model = model_data["model"]
    target_names = model_data["target_names"]
except FileNotFoundError:
    # Use a placeholder if the file is missing during initialization
    model = None
    target_names = []

@app.get("/")
def home():
    return {"message": "Iris Prediction API is running. Go to /docs for interactive testing."}

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_species(data: IrisRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found. Run training first.")

    # Format input for scikit-learn
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Make prediction
    prediction = model.predict(features)
    predicted_name = target_names[prediction[0]]

    return {
        "prediction": int(prediction[0]),
        "species": predicted_name
    }