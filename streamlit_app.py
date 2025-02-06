import streamlit as st
import requests

# API Endpoint
API_URL = "https://healthcare-llm-production.up.railway.app/predict"

st.title("🩺 Disease Prediction with AI")

symptoms = st.text_input("Enter symptoms (comma-separated, e.g., fever, cough, fatigue)")

if st.button("Predict Disease"):
    if symptoms:
        response = requests.post(API_URL, json={"symptoms": symptoms.split(",")})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"**Predicted Disease:** {result.get('predicted_disease', 'N/A')}")
            
            llm_explanation = result.get('llm_explanation', "⚠️ AI explanation not available.")
            st.info(f"**AI Explanation:** {llm_explanation}")
        else:
            st.error("⚠️ Failed to fetch prediction. Please try again later.")
    else:
        st.warning("Please enter at least one symptom.")
