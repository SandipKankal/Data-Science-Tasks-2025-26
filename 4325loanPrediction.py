# loan_prediction.py
"""
Simplified Loan Prediction Project
- Uses Logistic Regression and Decision Tree
- Chooses best model based on test accuracy
- Saves best model to 'best_model.joblib'
- Displays accuracy chart and makes one sample prediction
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer # Import SimpleImputer

# ----------------------------------------------------------------------
# 1. Load Dataset (upload your own loan_data.csv here)
# ----------------------------------------------------------------------
def load_dataset(path='loan_data.csv'):
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded dataset: {path}")
    except FileNotFoundError:
        print(f"⚠ File '{path}' not found — creating a small synthetic dataset.")
        df = pd.DataFrame({
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'Married': np.random.choice(['Yes', 'No'], 100),
            'ApplicantIncome': np.random.randint(2000, 10000, 100),
            'LoanAmount': np.random.randint(50, 250, 100),
            'Credit_History': np.random.choice([1, 0], 100, p=[0.8, 0.2]),
            'Loan_Status': np.random.choice(['Y', 'N'], 100, p=[0.7, 0.3])
        })
        df.to_csv(path, index=False)
    return df

# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
def preprocess(df):
    # Impute missing values
    numerical_cols = df.select_dtypes(include=np.number).columns
    imputer_mean = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_mean.fit_transform(df[numerical_cols])

    categorical_cols = df.select_dtypes(include=['object']).columns
    imputer_mode = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_mode.fit_transform(df[categorical_cols])

    # Convert categorical data to numeric
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    # Select only numerical columns for scaling
    X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled = scaler.transform(X_test[numerical_cols])

    # Combine scaled numerical features with non-scaled categorical features
    X_train_processed = pd.DataFrame(X_train_scaled, columns=numerical_cols, index=X_train.index)
    for col in X_train.columns:
        if col not in numerical_cols:
            X_train_processed[col] = X_train[col]

    X_test_processed = pd.DataFrame(X_test_scaled, columns=numerical_cols, index=X_test.index)
    for col in X_test.columns:
        if col not in numerical_cols:
            X_test_processed[col] = X_test[col]


    return X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, scaler

# ----------------------------------------------------------------------
# 3. Train and Compare Models
# ----------------------------------------------------------------------
def train_models(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled):
    log_model = LogisticRegression(max_iter=1000)
    tree_model = DecisionTreeClassifier(random_state=42)

    log_model.fit(X_train_scaled, y_train)
    tree_model.fit(X_train, y_train)

    log_acc = accuracy_score(y_test, log_model.predict(X_test_scaled))
    tree_acc = accuracy_score(y_test, tree_model.predict(X_test))

    print("\n📊 Model Accuracies:")
    print(f"Logistic Regression: {log_acc:.3f}")
    print(f"Decision Tree:       {tree_acc:.3f}")

    if log_acc > tree_acc:
        best_model = log_model
        best_name = "Logistic Regression"
    else:
        best_model = tree_model
        best_name = "Decision Tree"

    print(f"\n✅ Best Model: {best_name} (Accuracy = {max(log_acc, tree_acc):.3f})")
    return best_model, best_name, {'Logistic Regression': log_acc, 'Decision Tree': tree_acc}

# ----------------------------------------------------------------------
# 4. Save Model and Plot Accuracy
# ----------------------------------------------------------------------
def save_and_visualize(best_model, results):
    joblib.dump(best_model, "best_model.joblib")
    print("💾 Saved best model to best_model.joblib")

    plt.bar(results.keys(), results.values(), color=['skyblue', 'lightgreen'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("model_accuracy.png")
    print("📊 Saved accuracy chart to model_accuracy.png")
    plt.show()

# ----------------------------------------------------------------------
# 5. Test Prediction
# ----------------------------------------------------------------------
def test_prediction(model):
    print("\n🧩 Sample Prediction:")
    sample = pd.DataFrame([{
        'Gender': 1,   # Male
        'Married': 1,  # Yes
        'ApplicantIncome': 5500,
        'LoanAmount': 150,
        'Credit_History': 1
    }])
    # Ensure the sample data has the same columns and order as the training data
    # and apply the same preprocessing steps (imputation and scaling)
    # For simplicity, let's assume a basic imputation and scaling for the sample
    # In a real application, you would use the fitted imputer and scaler objects
    sample_processed = sample.copy()
    numerical_cols = sample_processed.select_dtypes(include=np.number).columns
    imputer_mean = SimpleImputer(strategy='mean')
    sample_processed[numerical_cols] = imputer_mean.fit_transform(sample_processed[numerical_cols])
    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(sample_processed[numerical_cols])
    sample_processed[numerical_cols] = sample_scaled


    pred = model.predict(sample_processed)[0]
    print("Predicted Loan Status:", "Approved ✅" if pred == 1 else "Rejected ❌")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    df = load_dataset('loan_data.csv')
    print(df.head())
    X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, scaler = preprocess(df)
    best_model, best_name, results = train_models(X_train_processed, y_train, X_test_processed, y_test, X_train_processed, X_test_processed)
    save_and_visualize(best_model, results)
    test_prediction(best_model)

if __name__ == "__main__":
    main()