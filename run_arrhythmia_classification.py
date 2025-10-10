# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import all the models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

print("--- Arrhythmia Classification Model Execution Started ---")

# Step 2: Load the dataset
# The dataset has no header and uses '?' for missing values
try:
    df = pd.read_csv('arrhythmia.data', header=None, na_values='?')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'arrhythmia.data' not found!")
    print("Please download the dataset from the UCI repository and place it in the same folder.")
    exit()

# Step 3: Data Cleaning and Preprocessing
print("⏳ Starting data cleaning and preprocessing...")

# Drop columns with a high percentage of missing values.
# This dataset has several columns that are almost entirely empty.
threshold = int(0.7 * len(df)) # Keep columns with at least 70% non-NA values
df = df.dropna(axis=1, thresh=threshold)

# For the remaining missing values, we will drop the rows that contain them
df = df.dropna(axis=0)

# The last column (226 after drops) is the target variable
target_column = df.columns[-1]

# Simplify the problem to binary classification: 1 (Normal) vs. 0 (Arrhythmia)
# The original dataset has Class 1 as 'normal' and 2-16 as different arrhythmias.
df['binary_target'] = np.where(df[target_column] == 1, 1, 0) # 1 for Normal, 0 for Arrhythmia
df = df.drop(target_column, axis=1) # Drop the original multi-class target

print("✅ Data cleaning complete. Problem simplified to Normal vs. Arrhythmia.")


# Step 4: Split data and apply feature scaling
X = df.drop('binary_target', axis=1)
y = df['binary_target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Data split. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Scale features for models that are sensitive to it (LogReg, SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 5: Train and Evaluate All Models
print("\n--- Model Training and Evaluation ---")

# Define the models to be evaluated
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine (SVM)": SVC(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier()
}

results = {}

# Loop through the models, train them, and store their accuracy
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"✅ {name}: Accuracy = {accuracy:.4f}")

# Step 6: Find and Announce the Best Model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print("\n--- Final Result ---")
print(f"🏆 Best Performing Model: {best_model_name} with an accuracy of {best_accuracy*100:.2f}%")
print("\n--- Execution Finished ---")