"""
app.py
------
Streamlit web app for diabetes risk prediction.
Trains the model on first run if artifacts are not found.

Usage:
    streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE  = "diabetes_model.joblib"
SCALER_FILE = "scaler.joblib"
DATA_FILE   = "diabetes.csv"
COLS_TO_CLEAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ── Train and cache model ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on first run — this takes ~30 seconds…")
def load_or_train():
    """Load saved artifacts if they exist, otherwise train from scratch."""
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)

    # Train from scratch
    df = pd.read_csv(DATA_FILE)
    for col in COLS_TO_CLEAN:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Best params found by GridSearchCV — hardcoded to avoid slow grid search on deploy
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Save for subsequent runs
    joblib.dump(model,  MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return model, scaler

model, scaler = load_or_train()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown(
    "This app predicts the likelihood of a patient having diabetes "
    "based on their medical data, using a tuned **Random Forest** classifier "
    "trained on the Pima Indians Diabetes Database."
)

st.sidebar.header("Patient Input Features")
st.sidebar.markdown("Adjust the values below, then press **Predict**.")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
def get_user_input() -> pd.DataFrame:
    pregnancies    = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose        = st.sidebar.number_input("Glucose (mg/dL)", 0, 200, 117)
    blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", 0, 122, 72)
    skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", 0, 99, 23)
    insulin        = st.sidebar.number_input("Insulin (μU/mL)", 0, 846, 30)
    bmi            = st.sidebar.number_input("BMI (kg/m²)", 0.0, 67.1, 32.0, step=0.1)
    dpf            = st.sidebar.number_input(
                       "Diabetes Pedigree Function", 0.078, 2.420, 0.3725, step=0.001,
                       help="Scores likelihood of diabetes based on family history."
                     )
    age            = st.sidebar.slider("Age (years)", 21, 81, 29)

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

# ── Main panel ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Patient Input Summary")
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
| Optimised for | Recall (sensitivity) |
| Test Accuracy | 74.03% |
| Recall (Diabetic) | 0.72 |
| Training data | Pima Indians Diabetes DB |
""")

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.sidebar.button("🔍 Predict", use_container_width=True):
    patient_scaled = scaler.transform(patient_df)
    prediction     = model.predict(patient_scaled)[0]
    proba          = model.predict_proba(patient_scaled)[0]
    confidence     = proba[prediction] * 100

    st.subheader("🔬 Prediction Result")

    r1, r2, r3 = st.columns(3)
    with r1:
        if prediction == 1:
            st.error("**⚠️ HIGH RISK — Diabetes Likely**")
        else:
            st.success("**✅ LOW RISK — No Diabetes Detected**")
    with r2:
        st.metric("Confidence", f"{confidence:.1f}%")
    with r3:
        st.metric("Risk Level",
            "High" if confidence > 70 else "Moderate" if confidence > 50 else "Low")

    st.markdown("**Prediction Probability Breakdown**")
    st.dataframe(pd.DataFrame({
        "Outcome":     ["No Diabetes", "Has Diabetes"],
        "Probability": [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
    }), use_container_width=True, hide_index=True)
    st.progress(float(proba[1]), text=f"Diabetes probability: {proba[1]*100:.1f}%")

    st.caption(
        "⚕️ **Disclaimer:** This tool is for educational purposes only and "
        "is not a substitute for professional medical advice, diagnosis, or treatment."
    )
else:
    st.info("👈 Adjust the patient values in the sidebar and press **Predict** to see results.")
