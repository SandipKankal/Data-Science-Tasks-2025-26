# -*- coding: utf-8 -*-
"""
Defaulter Prediction Model - Data Science Practical
---------------------------------------------------
This program builds a machine learning model to classify whether a person
is a 'defaulter' or 'non-defaulter' using a sample dataset.

It performs:
  - Data loading and preprocessing
  - Handling class imbalance using SMOTE
  - Model training (Random Forest)
  - Evaluation (Accuracy, AUC, Confusion Matrix, ROC Curve)
  - Feature importance visualization
  - Model saving and prediction on new samples
"""

# -------------------- 1) Import Required Libraries --------------------
import pandas as pd
import numpy as np
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------- 2) Download and Load Dataset --------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv"
filename = "defaulter_dataset.csv"

# Download the dataset only if it doesn't exist
if not os.path.exists(filename):
    print("Downloading dataset...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"✅ Dataset downloaded and saved as '{filename}'")
    else:
        raise Exception("❌ Failed to download dataset")

# Load dataset
df = pd.read_csv(filename, header=None)
print("\nDataset loaded successfully.")
print("Shape:", df.shape)

# -------------------- 3) Preprocess Target Column --------------------
# Assume last column is the target variable
df.rename(columns={df.columns[-1]: 'Defaulter_Status'}, inplace=True)
y = df['Defaulter_Status']
X = df.drop(columns=['Defaulter_Status'])

# Convert target labels to numeric (e.g., 'b'->0, 'g'->1)
target_mapping = {label: idx for idx, label in enumerate(y.unique())}
y = y.map(target_mapping)

print("\nTarget column encoded successfully.")
print("Target mapping:", target_mapping)

# -------------------- 4) Split Data and Handle Imbalance --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("\nApplied SMOTE. Class distribution after resampling:")
print(pd.Series(y_train_res).value_counts())

# -------------------- 5) Build and Train Random Forest Model --------------------
# Define pipeline: scaling + model
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ))
])

# Train model
rf_pipeline.fit(X_train_res, y_train_res)
print("\n✅ Random Forest model trained successfully!")

# -------------------- 6) Evaluate Model Performance --------------------
y_pred = rf_pipeline.predict(X_test)
y_pred_proba = rf_pipeline.predict_proba(X_test)[:, 1]

# Compute metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {acc:.3f}")
print(f"AUC Score: {auc:.3f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------- 7) Plot ROC Curve --------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Defaulter Prediction')
plt.legend()
plt.grid(True)
plt.show()

# -------------------- 8) Feature Importance Visualization --------------------
rf_model = rf_pipeline.named_steps['classifier']
feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_importance.head(15))
plt.title("Top 15 Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

# -------------------- 9) Save Model --------------------
joblib.dump(rf_pipeline, 'defaulter_rf_model.pkl')
print("\n💾 Model saved as 'defaulter_rf_model.pkl'")

# -------------------- 10) Predict on New Samples --------------------
print("\nPredictions for first 5 test samples:")
new_samples = X_test.head()
pred_probs = rf_pipeline.predict_proba(new_samples)[:, 1]
pred_classes = rf_pipeline.predict(new_samples)

for i, (cls, prob) in enumerate(zip(pred_classes, pred_probs), start=1):
    print(f"Sample {i}: Predicted class = {cls}, Probability = {prob:.3f}")
