from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

# Initialize FastAPI app
app = FastAPI()

# CORS settings to allow frontend to access the API
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model using Pydantic
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Load the saved logistic regression model with error handling
try:
    with open('LogisticRegression.sav', 'rb') as model_file:
        heart_disease_model = pickle.load(model_file)
except FileNotFoundError:
    raise RuntimeError("🚫 Model file 'LogisticRegression.sav' not found. Please ensure it's in the correct directory.")
except Exception as e:
    raise RuntimeError(f"🚫 Error loading model: {e}")

@app.get("/")
def read_root():
    return {"message": "🚀 Welcome to the Health Oracle API!"}

@app.post("/heart_disease_prediction")
def predict_heart_disease(input_parameters: HeartDiseaseInput):
    input_data = input_parameters.dict()

    # Extract features in the correct order
    features = [
        input_data['age'],
        input_data['sex'],
        input_data['cp'],
        input_data['trestbps'],
        input_data['chol'],
        input_data['fbs'],
        input_data['restecg'],
        input_data['thalach'],
        input_data['exang'],
        input_data['oldpeak'],
        input_data['slope'],
        input_data['ca'],
        input_data['thal']
    ]

    try:
        prediction = heart_disease_model.predict([features])
        result = "The person has heart disease." if prediction[0] == 1 else "The person does not have heart disease."
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
