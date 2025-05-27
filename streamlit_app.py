import streamlit as st
import pandas as pd
import numpy as np
import joblib


# 🛠 Page settings
st.set_page_config(page_title="CKD Risk Predictor", layout="centered")
st.title("🩺 Chronic Kidney Disease (CKD) Diagnostic app")
st.markdown("Estimate a patient’s CKD risk using lifestyle and clinical factors. 🚑")

# 🔄 Load models and scaler
xgb_model = joblib.load("xgboost_ckd_smote.pkl")
rf_model = joblib.load("random_forest_ckd_smote.pkl")
log_model = joblib.load("logistic_regression_ckd_smote_auc_0.9479.pkl")
scaler = joblib.load("scaler_ckd.pkl")

model_dict = {
    "XGBoost (AUC: 0.84)": xgb_model,
    "Random Forest (AUC: 0.85)": rf_model,
    "Logistic Regression (AUC: 0.74)": log_model
}

# 🧠 Sidebar model selector
model_choice = st.sidebar.selectbox("Choose a model", list(model_dict.keys()))
model = model_dict[model_choice]


# 🧾 Input form
st.markdown("### 🧪 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 45)
    gender = st.selectbox("Sex", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["Black", "White", "Asian", "Hispanic", "Other"])
    education = st.selectbox("Education Level", ["No education", "Primary", "Secondary", "Tertiary"])
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

# Clinical metrics
st.markdown("### 🧾 Clinical Measurements")
SBP = st.slider("Systolic Blood Pressure (SBP)", 90, 200, 130)
DBP = st.slider("Diastolic Blood Pressure (DBP)", 60, 120, 80)
BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=24.0)
heart_rate = st.slider("Heart Rate", 40, 150, 75)

# 🔢 Encode inputs
def encode_binary(val): return 1 if val == "Yes" else 0

encoded_input = pd.DataFrame([[
    1 if gender == "Male" else 0,
    {"No education": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}[education],
    {"Low": 0, "Middle": 1, "High": 2}[socio],
    encode_binary(hypertension),
    encode_binary(diabetes),
    encode_binary(cardiovascular),
    encode_binary(family_history),
    encode_binary(smoking),
    encode_binary(alcohol),
    encode_binary(fatigue),
    encode_binary(swelling),
    encode_binary(urinary_freq),
    encode_binary(nocturia),
    encode_binary(hematuria),
    encode_binary(nausea),
    encode_binary(muscle_cramps),
    encode_binary(pallor),
    encode_binary(herbal_meds),
    {"Good": 1, "Poor": 0}[healthcare],
    SBP,
    DBP,
    BMI,
    heart_rate
]], columns=[
    'Gender_Male', 'Education_Level', 'Socioeconomic_Status', 'Hypertension', 'Diabetes',
    'Cardiovascular_Disease', 'Family_History', 'Smoking', 'Alcohol', 'Fatigue',
    'Swelling', 'Urinary_Frequency', 'Nocturia', 'Hematuria', 'Nausea',
    'Muscle_Cramps', 'Pallor', 'Herbal_Medication_Use', 'Healthcare_Access',
    'SBP', 'DBP', 'BMI', 'Heart_Rate'
])

# 🚦 Predict button
if st.button("🔍 Diagnose CKD"):
    if "Logistic" in model_choice:
        input_scaled = scaler.transform(encoded_input)
        prob = model.predict_proba(input_scaled)[0][1]
    else:
        prob = model.predict_proba(encoded_input)[0][1]

    st.markdown(f"<h2 style='color:#EF476F;'>CKD Risk: {prob:.2%}</h2>", unsafe_allow_html=True)
    st.info(f"🔢 Model: {model_choice.split()[0]} | AUC: {model_choice.split(': ')[-1]}")

    if prob > 0.5:
        st.error("⚠️ High Risk of CKD. Recommend clinical evaluation.")
    else:
        st.success("✅ Low Risk. Continue preventive care and lifestyle management.")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Part of the NCD Risk Suite • Built by Oluwafemi</p>", unsafe_allow_html=True)