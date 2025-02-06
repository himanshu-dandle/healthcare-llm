import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
import openai
import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("❌ OpenAI API Key not found. Please check your `.env` file or set it manually.")

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
MODEL_PATH = os.path.join(BASE_DIR, "../models/disease_prediction_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Load the symptoms list
SYMPTOMS_PATH = os.path.join(BASE_DIR, "../data/processed/disease_symptom_encoded_augmented.csv")
if not os.path.exists(SYMPTOMS_PATH):
    raise FileNotFoundError(f"❌ Symptoms dataset not found: {SYMPTOMS_PATH}")
df = pd.read_csv(SYMPTOMS_PATH)
symptom_columns = df.columns[1:]  # Exclude "Disease" column

# Initialize FastAPI
app = FastAPI(title="Disease Prediction API")

# Define input data structure
class SymptomsRequest(BaseModel):
    symptoms: list[str]

def query_openai(disease: str):
    """Query OpenAI API for disease explanation."""
    prompt = f"Explain the disease {disease} in simple terms, including its symptoms, causes, and treatment."

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error fetching explanation: {str(e)}"

@app.post("/predict")
def predict_disease(request: SymptomsRequest):
    """Predict disease based on symptoms and return LLM explanation."""
    dataset_symptoms = [col.lower().replace(" ", "_") for col in symptom_columns]
    input_vector = pd.DataFrame([[0] * len(symptom_columns)], columns=symptom_columns)

    recognized_symptoms = []
    unrecognized_symptoms = []

    for symptom in request.symptoms:
        symptom_cleaned = symptom.lower().replace(" ", "_")
        if symptom_cleaned in dataset_symptoms:
            matched_column = symptom_columns[dataset_symptoms.index(symptom_cleaned)]
            input_vector[matched_column] = 1
            recognized_symptoms.append(matched_column)
        else:
            unrecognized_symptoms.append(symptom)

    # If no symptoms were recognized, return an error
    if not recognized_symptoms:
        raise HTTPException(status_code=400, detail="❌ No valid symptoms recognized. Please enter correct symptoms.")

    print(f"✅ Recognized Symptoms: {recognized_symptoms}")
    print(f"❌ Unrecognized Symptoms: {unrecognized_symptoms}")
    print("🔹 Input Symptom Vector:\n", input_vector)

    predicted_disease = model.predict(input_vector)[0]
    llm_explanation = query_openai(predicted_disease)

    return {
        "predicted_disease": predicted_disease,
        "recognized_symptoms": recognized_symptoms,
        "unrecognized_symptoms": unrecognized_symptoms,
        "llm_explanation": llm_explanation
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
