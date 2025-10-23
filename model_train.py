import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib

def visualize_data(datafame):
    sns.set_style('whitegrid')

    print("\n--- Generating Plots ---")

    df.hist(bins=15,figsize=(15,10))
    plt.suptitle("Feature Distributions")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Correlation Matrix of Features')
    plt.show()
    print("\n--- Analysis Complete ---")
    print("Check the plots to see the data distributions and correlations.")



try:
    df=pd.read_csv("diabetes.csv")

except FileNotFoundError:
    print(f"Error: The file '{"diabetes.csv"}' was not found")
    print("Please make sure you have downloaded it and placed it in the correct directory.")

cols_to_clean = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for col in cols_to_clean:
    df[col]= df[col].replace(0,np.nan)

for col in cols_to_clean:
    median_val=df[col].median()
    df[col].fillna(median_val,inplace=True)



print("--Data Cleaned Successfully--")
print("Zeros have been replaced with the median value in key columns.")
print("\n## New Statistical Summary:")
print(df.describe())

#define features X and Y, X  will contain all data except the Outcome
#Y will be the outcome
X=df.drop('Outcome',axis=1)
y=df['Outcome']

#Data splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
scaler = StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model_base=RandomForestClassifier(random_state=42,class_weight="balanced")

param_grid= {
    'n_estimators':[100,200,300],
    'max_depth':[5,10,15],
    'min_samples_leaf':[1,2,4]

}
grid_search=GridSearchCV(estimator=model_base,param_grid=param_grid,cv=5,scoring='recall',n_jobs=-1,verbose=1)
grid_search.fit(X_train_scaled,y_train)
best_model=grid_search.best_estimator_
print("\n--- Grid search Complete! ---")
print(f"best parameters found: {grid_search.best_params_}")


#evaluation of model
y_pred=best_model.predict(X_test_scaled)
accuracy=accuracy_score(y_test,y_pred)
print(f"model accuracy: {accuracy * 100:.2f}%")
print("-"*35)

print("classifcation report:")
print(classification_report(y_test,y_pred))
print("-"*35)

print("confusion matrix:")

cm= confusion_matrix(y_test,y_pred)
print(cm)
print("-"*35)
print("\nEvaluation complete. The results above show how well the model performed.")



#Saving the model
print("\n--- Saving model and scaler to files ---")
joblib.dump(best_model,'diabetes_model.joblib')
joblib.dump(scaler,'scaler.joblib')
print("Model and scaler saved successfully as 'diabetes_model.joblib' and 'scaler.joblib'")