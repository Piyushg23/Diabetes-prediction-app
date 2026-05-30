# 🩺 Diabetes Risk Predictor

A machine learning web app that predicts a patient's diabetes risk from medical data.  
Built end-to-end — from data cleaning to a live interactive UI.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📸 Overview

The app takes 8 medical inputs (glucose, BMI, age, etc.) and returns a real-time diabetes risk prediction with a confidence score, powered by a tuned **Random Forest** classifier.

---

## 🔬 What I Built

### Data Cleaning
The Pima Indians Diabetes Database uses `0` for missing values in columns where zero is physiologically impossible (e.g. BMI, Glucose, Blood Pressure). I replaced all zeros in these columns with the **column median** to preserve the distribution without dropping rows.

### Modelling
Trained a **Random Forest Classifier** — a strong ensemble method well-suited to tabular medical data with mixed feature importance.

### Tuning
Used **GridSearchCV** with 5-fold cross-validation, optimising for **recall** — the priority here is catching diabetic patients (true positives), even at the cost of some false alarms.

```
Tuned parameters:
  n_estimators:    [100, 200, 300]
  max_depth:       [5, 10, 15]
  min_samples_leaf:[1, 2, 4]
```

### Deployment
Saved the tuned model and scaler with `joblib`, then built an interactive Streamlit UI to serve live predictions.

---

## 📊 Model Performance

Evaluated on an unseen 20% test split:

| Metric | Value |
|--------|-------|
| Accuracy | 74.03% |
| Recall (Diabetic class) | 0.72 |
| Precision (Diabetic class) | 0.61 |

**Confusion Matrix:**
```
              Predicted
              No   Yes
Actual  No  [ 75   25 ]
        Yes [ 15   39 ]
```
The model correctly identified **39 of 54 diabetic patients** in the test set.

---

## 🚀 Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-prediction-app.git
cd diabetes-prediction-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model** *(skip if you use the pre-trained files)*
```bash
python train_model.py
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
diabetes-prediction-app/
├── app.py               # Streamlit web UI
├── train_model.py       # Data cleaning, training, evaluation, saving
├── diabetes.csv         # Pima Indians Diabetes Dataset
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** `diabetes_model.joblib` and `scaler.joblib` are excluded from the repo via `.gitignore`.  
> Run `python train_model.py` to generate them locally.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Pandas / NumPy | Data cleaning and manipulation |
| Scikit-learn | Model training, scaling, GridSearchCV |
| Joblib | Model serialisation |
| Streamlit | Interactive web UI |
| Matplotlib / Seaborn | EDA visualisations |

---

## ⚕️ Disclaimer

This tool is for **educational purposes only** and is not a substitute for professional medical advice, diagnosis, or treatment.
