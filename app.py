import streamlit as st
import requests

# API URL (Change this if your API is deployed)

##API_URL = "http://127.0.0.1:8000/predict"

API_URL = "http://127.0.0.1:8000/predict"



# Streamlit UI
st.title("🩺 Symptom-based Disease Prediction")

# Input for symptoms
symptoms = st.text_input("Enter symptoms (comma-separated)", placeholder="e.g., fever, headache, fatigue")

if st.button("Predict Disease"):
    if symptoms:
        # Send request to FastAPI backend
        response = requests.post(API_URL, json={"symptoms": symptoms.split(",")})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"🩺 **Predicted Disease:** {result['predicted_disease']}")
            st.markdown("### 📖 LLM Explanation:")
            st.write(result["llm_explanation"])
        else:
            st.error("⚠️ Error fetching response from API.")
    else:
        st.warning("⚠️ Please enter symptoms.")

