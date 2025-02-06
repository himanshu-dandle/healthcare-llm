import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
import openai
import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if key is loaded
if openai.api_key is None:
    raise ValueError("OpenAI API Key not found. Please check your `.env` file or set it manually.")

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
MODEL_PATH = os.path.join(BASE_DIR, "../models/disease_prediction_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Load the symptoms list
SYMPTOMS_PATH = os.path.join(BASE_DIR, "../data/processed/disease_symptom_encoded_augmented.csv")
if not os.path.exists(SYMPTOMS_PATH):
    raise FileNotFoundError(f"Symptoms dataset not found: {SYMPTOMS_PATH}")
df = pd.read_csv(SYMPTOMS_PATH)
symptom_columns = df.columns[1:]  # Exclude the "Disease" column

# Initialize FastAPI
app = FastAPI(title="Disease Prediction API")

# Define input data structure
class SymptomsRequest(BaseModel):
    symptoms: list[str]

@app.post("/predict")
def predict_disease(request: SymptomsRequest):
    """Predict disease based on symptoms."""
    input_vector = pd.DataFrame([[0] * len(symptom_columns)], columns=symptom_columns)
    
    recognized_symptoms = []
    unrecognized_symptoms = []
    
    dataset_symptoms = [col.lower().replace(" ", "_") for col in symptom_columns]

    for symptom in request.symptoms:
        symptom_cleaned = symptom.lower().replace(" ", "_")
        if symptom_cleaned in dataset_symptoms:
            matched_column = symptom_columns[dataset_symptoms.index(symptom_cleaned)]
            input_vector[matched_column] = 1
            recognized_symptoms.append(matched_column)
        else:
            unrecognized_symptoms.append(symptom)

    predicted_disease = model.predict(input_vector)[0]

    return {
        "predicted_disease": predicted_disease,
        "recognized_symptoms": recognized_symptoms,
        "unrecognized_symptoms": unrecognized_symptoms
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Railway provides PORT automatically
    uvicorn.run(app, host="0.0.0.0", port=port)
