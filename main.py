from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('model/model.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Create an instance of FastAPI
app = FastAPI()

# Define the request body for input data (features or independent variables)
class PredictRequest(BaseModel):
    BUYING_PRIZE: float

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Access the input data from the request object
        data = np.array([[request.BUYING_PRIZE]])

        # Get the prediction from the model
        prediction = model.predict(data)

        # Return the prediction in a response
        return {
            "input": {"If Buying Price is: ": request.BUYING_PRIZE},
            "Your selling Price should be: ": prediction[0]
        }

    except Exception as e:
        # Handle errors gracefully and return an HTTP 400 Bad Request
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Linear Regression API"}
