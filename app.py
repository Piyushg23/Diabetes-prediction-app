"""
app.py
------
Streamlit web app for diabetes risk prediction.
Loads the pre-trained Random Forest model and scaler saved by train_model.py.

Usage:
    streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("diabetes_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found. "
        "Please run `python train_model.py` first to generate them."
    )
    st.stop()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown(
    "This app predicts the likelihood of a patient having diabetes "
    "based on their medical data, using a tuned **Random Forest** classifier "
    "trained on the Pima Indians Diabetes Database."
)

st.sidebar.header("Patient Input Features")
st.sidebar.markdown("Adjust the sliders and inputs, then press **Predict**.")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
def get_user_input() -> pd.DataFrame:
    pregnancies     = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose         = st.sidebar.number_input("Glucose (mg/dL)", 0, 200, 117)
    blood_pressure  = st.sidebar.number_input("Blood Pressure (mm Hg)", 0, 122, 72)
    skin_thickness  = st.sidebar.number_input("Skin Thickness (mm)", 0, 99, 23)
    insulin         = st.sidebar.number_input("Insulin (μU/mL)", 0, 846, 30)
    bmi             = st.sidebar.number_input("BMI (kg/m²)", 0.0, 67.1, 32.0, step=0.1)
    dpf             = st.sidebar.number_input(
                        "Diabetes Pedigree Function", 0.078, 2.420, 0.3725, step=0.001,
                        help="A function that scores likelihood of diabetes based on family history."
                      )
    age             = st.sidebar.slider("Age (years)", 21, 81, 29)

    return pd.DataFrame([{
        "Pregnancies":              pregnancies,
        "Glucose":                  glucose,
        "BloodPressure":            blood_pressure,
        "SkinThickness":            skin_thickness,
        "Insulin":                  insulin,
        "BMI":                      bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age":                      age,
    }])

patient_df = get_user_input()

# ── Main panel: input summary ─────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Patient Input Summary")
    # Friendly display table
    display = patient_df.T.rename(columns={0: "Value"})
    display.index = [
        "Pregnancies", "Glucose (mg/dL)", "Blood Pressure (mm Hg)",
        "Skin Thickness (mm)", "Insulin (μU/mL)", "BMI (kg/m²)",
        "Diabetes Pedigree Function", "Age (years)"
    ]
    st.dataframe(display, use_container_width=True)

with col2:
    st.subheader("📊 Model Information")
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | Algorithm | Random Forest |
    | Tuning | GridSearchCV (5-fold CV) |
    | Optimised for | Recall (sensitivity) |
    | Test Accuracy | 74.03% |
    | Recall (Diabetic) | 0.72 |
    | Training data | Pima Indians Diabetes DB |
    """)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
predict_btn = st.sidebar.button("🔍 Predict", use_container_width=True)

if predict_btn:
    patient_scaled = scaler.transform(patient_df)
    prediction     = model.predict(patient_scaled)[0]
    proba          = model.predict_proba(patient_scaled)[0]
    confidence     = proba[prediction] * 100

    st.subheader("🔬 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error("**⚠️ HIGH RISK — Diabetes Likely**")
        else:
            st.success("**✅ LOW RISK — No Diabetes Detected**")

    with res_col2:
        st.metric("Confidence", f"{confidence:.1f}%")

    with res_col3:
        st.metric(
            "Risk Level",
            "High" if confidence > 70 else "Moderate" if confidence > 50 else "Low"
        )

    # Probability breakdown
    st.markdown("**Prediction Probability Breakdown**")
    prob_df = pd.DataFrame({
        "Outcome":     ["No Diabetes", "Has Diabetes"],
        "Probability": [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
        "Bar":         [proba[0], proba[1]],
    })
    st.dataframe(prob_df[["Outcome", "Probability"]], use_container_width=True, hide_index=True)
    st.progress(float(proba[1]), text=f"Diabetes probability: {proba[1]*100:.1f}%")

    st.caption(
        "⚕️ **Disclaimer:** This tool is for educational purposes only and "
        "is not a substitute for professional medical advice, diagnosis, or treatment."
    )
else:
    st.info("👈 Adjust the patient values in the sidebar and press **Predict** to see results.")
