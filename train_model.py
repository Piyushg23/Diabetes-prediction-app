"""
train_model.py
--------------
Trains a Random Forest classifier on the Pima Indians Diabetes Database.
Saves the tuned model and scaler to disk for use by app.py.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_FILE  = "diabetes.csv"
MODEL_FILE = "diabetes_model.joblib"
SCALER_FILE = "scaler.joblib"

COLS_TO_CLEAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# ── 1. Load ───────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"'{DATA_FILE}' not found. "
        "Download the Pima Indians Diabetes Dataset and place it here."
    )

print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Clean — replace impossible zeros with column median ────────────────────
for col in COLS_TO_CLEAN:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

print("Data cleaned — zeros replaced with column medians in:", COLS_TO_CLEAN)

# ── 3. Optional EDA plots ─────────────────────────────────────────────────────
def visualize_data(dataframe: pd.DataFrame) -> None:
    """Generate and display EDA plots."""
    sns.set_style("whitegrid")

    dataframe.hist(bins=15, figsize=(15, 10))
    plt.suptitle("Feature Distributions")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("feature_distributions.png", dpi=100)
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(dataframe.corr(), annot=True, cmap="viridis", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=100)
    plt.show()
    print("EDA plots saved: feature_distributions.png, correlation_matrix.png")

# Uncomment to generate plots:
# visualize_data(df)

# ── 4. Split ──────────────────────────────────────────────────────────────────
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── 5. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 6. Train with GridSearchCV (optimising for recall) ───────────────────────
param_grid = {
    "n_estimators":   [100, 200, 300],
    "max_depth":      [5, 10, 15],
    "min_samples_leaf": [1, 2, 4],
}

base_model = RandomForestClassifier(random_state=42, class_weight="balanced")
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = best_model.predict(X_test_scaled)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Has Diabetes"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Has Diabetes"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
print("Confusion matrix saved: confusion_matrix.png")

# ── 8. Save ───────────────────────────────────────────────────────────────────
joblib.dump(best_model, MODEL_FILE)
joblib.dump(scaler,     SCALER_FILE)
print(f"\nSaved: {MODEL_FILE}, {SCALER_FILE}")
