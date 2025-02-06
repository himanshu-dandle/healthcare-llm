import openai
import os
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the trained disease prediction model
model_path = "models/disease_prediction_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"⚠️ Model not found at {model_path}. Train the model first!")

model = joblib.load(model_path)

# Load processed dataset for symptom reference
df = pd.read_csv("data/processed/disease_symptom_encoded_augmented.csv")
symptom_names = df.columns[1:]  # Exclude 'Disease' column

# Function to query OpenAI for detailed explanation
def query_openai(disease_name):
    print("Querying OpenAI...")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Explain the disease {disease_name} in detail, including symptoms, causes, and treatment."}
        ]
    )
    return response.choices[0].message.content.strip()

# Main prediction workflow
print("\n🔹 Symptom-based Disease Prediction using OpenAI 🔹")
user_input = input("Enter symptoms separated by commas (e.g., fever, headache, fatigue): ").lower().split(",")

# Match user input symptoms with available symptoms
matched_symptoms = [sym.strip() for sym in user_input if sym.strip() in symptom_names]

if not matched_symptoms:
    print("⚠️ No valid symptoms recognized. Please enter symptoms from the dataset.")
    exit()

# Convert symptoms into model input format
input_features = np.zeros(len(symptom_names))  # Initialize zero array
for i, symptom in enumerate(symptom_names):
    if symptom in matched_symptoms:
        input_features[i] = 1  # Set symptom presence to 1

# Convert input to DataFrame format for prediction
input_df = pd.DataFrame([input_features], columns=symptom_names)

# Predict the disease using the trained model
predicted_disease = model.predict(input_df)[0]
print(f"\n🩺 Predicted Disease: {predicted_disease}")

# Get detailed information from OpenAI
disease_info = query_openai(predicted_disease)
print(f"\n📖 LLM Explanation:\n{disease_info}")
