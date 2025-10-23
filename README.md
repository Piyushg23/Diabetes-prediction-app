# Diabetes-prediction-app
This is a simple machine learning project I built to practice the full ML workflow. It's a web app that predicts a patient's diabetes risk based on their medical data.

I used Scikit-learn for the model and Streamlit to build the simple user interface.

What I Did
The goal was to build a working prediction tool from scratch, not just a model.

Data Cleaning: I started with the Pima Indians Diabetes Database. The first step was cleaning it up, as it used '0' for missing values in columns like 'BMI' or 'Glucose'. I replaced all these impossible zeros with the median value for that column.

Modeling: I trained a Random Forest Classifier since it's a strong all-around model.

Tuning: I used GridSearchCV to find the best settings for the model. My main goal was to improve recall, since it's more important to correctly identify a sick patient (even if it means a few false alarms) than to miss a case.

Deployment: I saved the final, tuned model and the data scaler using joblib. Then, I built the app.py script to load those files and serve a live, interactive app.

Model Performance
After tuning, the final Random Forest model had these results on the unseen test data:

Accuracy: 74.03%

Recall (for 'Has Diabetes'): 0.72 (Successfully found 72% of all diabetic patients)

Confusion Matrix:
[[75 25]
 [15 39]]
(This means it correctly identified 39 diabetic patients while missing 15).

How to Run It
Clone this repository.

Install the needed libraries:
pip install -r requirements.txt

(If you want to re-train) Run the training script:
python train_model.py

Run the web app:
streamlit run app.py
