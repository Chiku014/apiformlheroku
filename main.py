from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

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

# Define the input data model
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

# Load the saved logistic regression model
with open('LogisticRegression.sav', 'rb') as model_file:
    heart_disease_model = pickle.load(model_file)

@app.post('/heart_disease_prediction')
def predict_heart_disease(input_parameters: HeartDiseaseInput):
    input_data = input_parameters.dict()
    
    # Extract features from the input data
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

    # Make prediction
    prediction = heart_disease_model.predict([features])

    # Return prediction result
    return {
        "prediction": "The person has heart disease." if prediction[0] == 1 else "The person does not have heart disease."
    }
@app.get("/")
def read_root():
    return {"message": "Welcome to the Health Oracle API!"}