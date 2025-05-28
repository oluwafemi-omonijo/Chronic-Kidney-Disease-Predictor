import streamlit as st
import pandas as pd
import joblib

# 🛠 Page Configuration
st.set_page_config(page_title="CKD Risk Predictor", layout="centered")
st.title("🩺 Chronic Kidney Disease (CKD) Risk Predictor")
st.markdown("Estimate a patient's risk of CKD using clinical and lifestyle information. 🚑")

# 🔄 Load Model
model = joblib.load("random_forest_ckd_model.pkl")

# 🧾 Patient Inputs
st.markdown("## 🔎 Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 45)
    gender = st.selectbox("Sex", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["Black", "White", "Asian", "Hispanic", "Other"])
    socio = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    cardiovascular = st.selectbox("Cardiovascular Disease", ["Yes", "No"])
    family_history = st.selectbox("Family History of CKD", ["Yes", "No"])
    healthcare = st.selectbox("Access to Healthcare", ["Good", "Poor"])

with col2:
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Intake", ["Yes", "No"])
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    swelling = st.selectbox("Swelling (Edema)", ["Yes", "No"])
    urinary_freq = st.selectbox("Urinary Frequency", ["Yes", "No"])
    nocturia = st.selectbox("Nocturia", ["Yes", "No"])
    hematuria = st.selectbox("Hematuria", ["Yes", "No"])
    nausea = st.selectbox("Nausea", ["Yes", "No"])
    muscle_cramps = st.selectbox("Muscle Cramps", ["Yes", "No"])
    pallor = st.selectbox("Pallor", ["Yes", "No"])
    herbal_meds = st.selectbox("Herbal Medication Use", ["Yes", "No"])

# Clinical measurements
st.markdown("## 🧪 Clinical Measurements")
SBP = st.slider("Systolic BP (SBP)", 90, 200, 130)
DBP = st.slider("Diastolic BP (DBP)", 60, 120, 80)
BMI = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 24.0)
heart_rate = st.slider("Heart Rate", 40, 150, 75)

# 🧠 Encode Inputs
def encode(val): return 1 if val == "Yes" else 0
ethnicity_map = {
    "Black": [1, 0, 0],
    "White": [0, 1, 0],
    "Other": [0, 0, 1],
    "Asian": [0, 0, 1],
    "Hispanic": [0, 0, 1]
}
eth_encoded = ethnicity_map[ethnicity]

input_df = pd.DataFrame([[
    age,
    1 if gender == "Male" else 0,
    {"Low": 0, "Middle": 1, "High": 2}[socio],
    encode(hypertension), encode(diabetes), encode(cardiovascular),
    encode(family_history), encode(smoking), encode(alcohol),
    encode(fatigue), encode(swelling), encode(urinary_freq),
    encode(nocturia), encode(hematuria), encode(nausea),
    encode(muscle_cramps), encode(pallor), encode(herbal_meds),
    1 if healthcare == "Good" else 0,
    SBP, DBP, BMI, heart_rate,
    *eth_encoded
]], columns=[
    'Age', 'Gender_Male', 'Socioeconomic_Status', 'Hypertension', 'Diabetes',
    'Cardiovascular_Disease', 'Family_History', 'Smoking', 'Alcohol',
    'Fatigue', 'Swelling', 'Urinary_Frequency', 'Nocturia', 'Hematuria',
    'Nausea', 'Muscle_Cramps', 'Pallor', 'Herbal_Medication_Use',
    'Healthcare_Access', 'SBP', 'DBP', 'BMI', 'Heart_Rate',
    'Ethnicity_Black', 'Ethnicity_White', 'Ethnicity_Other'
])

# 🚀 Prediction
if st.button("🔍 Predict CKD Risk"):
    prob = model.predict_proba(input_df)[0][1]
    st.markdown(f"<h2 style='color:#EF476F;'>CKD Risk: {prob:.2%}</h2>", unsafe_allow_html=True)
    st.info("Model: Random Forest (AUC: 0.85)")

    if prob > 0.5:
        st.error("⚠️ High Risk of CKD. Recommend clinical evaluation.")
    else:
        st.success("✅ Low Risk. Continue preventive care and lifestyle management.")

# Footer
st.markdown("---")
st.markdown("<center><sub>Part of the NCD Risk Suite • Built by Oluwafemi</sub></center>", unsafe_allow_html=True)
