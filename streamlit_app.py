import streamlit as st
import requests

# API Endpoint
##API_URL = "http://127.0.0.1:8000/predict"
API_URL = "https://healthcare-llm-production.up.railway.app/predict"


# Streamlit UI
st.title("🩺 Disease Prediction using Symptoms")

# Input field for symptoms
symptoms = st.text_input("Enter symptoms separated by commas (e.g., fever, headache, fatigue)")

# Predict button
if st.button("Predict Disease"):
    if symptoms:
        # Send request to FastAPI backend
        response = requests.post(API_URL, json={"symptoms": symptoms.split(",")})

        if response.status_code == 200:
            result = response.json()
            st.success(f"**Predicted Disease:** {result['predicted_disease']}")
            st.info(f"**AI Explanation:** {result['llm_explanation']}")
        else:
            st.error("Error occurred while fetching prediction. Please check the API.")
    else:
        st.warning("⚠️ Please enter symptoms to proceed.")

# Run the Streamlit app using:
# `streamlit run streamlit_app.py`
