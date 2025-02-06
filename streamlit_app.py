import streamlit as st
import requests
import os

# Detect if running in a cloud environment
if "RAILWAY_ENVIRONMENT" in os.environ:
    API_URL = "https://healthcare-llm-production.up.railway.app/predict"  # Cloud Deployment
else:
    API_URL = "http://127.0.0.1:8000/predict"  # Local Development

st.title("Disease Prediction using LLM & ML")

symptoms = st.text_input("Enter symptoms separated by commas (e.g., fever, headache, fatigue):")

if st.button("Predict Disease"):
    if symptoms:
        try:
            response = requests.post(API_URL, json={"symptoms": symptoms.split(",")})
            result = response.json()

            st.success(f"**Predicted Disease:** {result.get('predicted_disease', 'Unknown')}")
            
            # Handle missing LLM explanation
            llm_explanation = result.get("llm_explanation", "⚠️ AI explanation not available.")
            st.info(f"**AI Explanation:** {llm_explanation}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Ensure FastAPI is running locally or check deployment.")
    else:
        st.warning("⚠️ Please enter symptoms to get a prediction.")
