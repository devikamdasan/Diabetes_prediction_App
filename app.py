import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn and other warnings

import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

@st.cache_resource
def load_model_scaler():
    model = load_model("diabetes_ann_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler
ann, scaler = load_model_scaler()
st.title("Diabetes Prediction using ANN")
st.write("Enter patient information below:")
year = st.number_input("Year", min_value=1900, max_value=2100, value=2026)
gender = st.selectbox("Gender", options=["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
location = st.selectbox("Location", options=["Urban", "Rural"])
race_AfricanAmerican = st.number_input("Race: African American", min_value=0, max_value=1, value=0)
race_Asian = st.number_input("Race: Asian", min_value=0, max_value=1, value=0)
race_Caucasian = st.number_input("Race: Caucasian", min_value=0, max_value=1, value=1)
race_Hispanic = st.number_input("Race: Hispanic", min_value=0, max_value=1, value=0)
race_Other = st.number_input("Race: Other", min_value=0, max_value=1, value=0)
hypertension = st.selectbox("Hypertension", options=[0, 1])
heart_disease = st.selectbox("Heart Disease", options=[0, 1])
smoking_history = st.selectbox("Smoking History", options=["never", "formerly", "current", "unknown"])
bmi = st.number_input("BMI", value=25.0)
hbA1c_level = st.number_input("HbA1c Level", value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level", value=90.0)
gender_enc = 0 if gender == "Male" else 1
location_enc = 0 if location == "Urban" else 1
smoking_mapping = {"never":0, "formerly":1, "current":2, "unknown":3}
smoking_enc = smoking_mapping[smoking_history]


if st.button("Predict Diabetes"):
    input_data = np.array([[year, gender_enc, age, location_enc, 
                            race_AfricanAmerican, race_Asian, race_Caucasian, race_Hispanic, race_Other,
                            hypertension, heart_disease, smoking_enc, bmi, hbA1c_level, blood_glucose_level]])
    input_scaled = scaler.transform(input_data)
    prediction = ann.predict(input_scaled)
    result = "Diabetes" if prediction[0][0] > 0.5 else "No Diabetes"
    
    st.success(f"Prediction: {result}")

st.markdown("""
---
**Disclaimer:** This prediction is for educational purposes only.  
It does not replace professional medical advice.
""")