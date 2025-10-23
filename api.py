import streamlit as st
import joblib
import numpy as np
try:
    model=joblib.load('diabetes_model.joblib')
    scaler=joblib.load('scaler.joblib')


except FileNotFoundError:
    st.error("Model or scaler not found, please run the fucking training script first")
    st.stop()

#building web interface
st.title('early diabetes prediction')
st.write("This app predicts the likelihood of a patient having diabetes based on their medical information.")
st.sidebar.header('Patient Input Features')

def user_input_features():
    pregnancies= st.sidebar.slider('Pregnancies',0,17,3)
    glucose=st.sidebar.number_input('Glucose',min_value=0,max_value=200,value=117)
    blood_pressure=st.sidebar.number_input('Blood Pressure (mm Hg)',min_value=0,max_value=122,value=72)
    Skin_thickness=st.sidebar.number_input('Skin Thickness(mm)',min_value=0,max_value=99,value=23)
    insulin = st.sidebar.number_input('Insulin (mu U/ml)', min_value=0, max_value=846, value=30)
    bmi = st.sidebar.number_input('BMI (weight in kg/(height in m)^2)', min_value=0.0, max_value=67.1, value=32.0)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.3725)
    age=st.sidebar.slider('Age(years)',21,81,29)



    data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': Skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
    



    features=np.array([list(data.values())])
    return features
patient_data=user_input_features()

st.subheader('Patient Input: ')
st.write(patient_data)

if st.sidebar.button('Predict'):
    patient_data_scaled=scaler.transform(patient_data)

    prediction=model.predict(patient_data_scaled)
    prediction_proba=model.predict_proba(patient_data_scaled)


    st.subheader('Prediction Result')
    if prediction[0]==1:
        st.warning('The model predicts: HAS DIABETES')
    else:
        st.success('The model predicts: NO DIABETES')

    st.subheader('Prediction Probability')
    st.write(f"Confidence: **{prediction_proba[0][prediction[0]] * 100:.2f}%**")