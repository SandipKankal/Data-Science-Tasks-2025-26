# loan_prediction.py
"""
Loan Prediction project
- Trains Logistic Regression and Decision Tree
- Picks the best model by test accuracy
- Saves the best model pipeline to best_model.joblib
- Demonstrates loading and predicting with saved model
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

RND = 42

def create_synthetic_dataset(path='loan_data.csv', n=500, random_state=RND):
    """
    Creates a realistic synthetic Loan Prediction dataset and saves to CSV (if file not present).
    """
    rng = np.random.RandomState(random_state)
    Gender = rng.choice(['Male','Female'], size=n, p=[0.7,0.3])
    Married = rng.choice(['Yes','No'], size=n, p=[0.7,0.3])
    Dependents = rng.choice(['0','1','2','3+'], size=n, p=[0.5,0.2,0.15,0.15])
    Education = rng.choice(['Graduate','Not Graduate'], size=n, p=[0.85,0.15])
    Self_Employed = rng.choice(['Yes','No'], size=n, p=[0.1,0.9])
    ApplicantIncome = rng.normal(5000, 2000, size=n).clip(1000)
    CoapplicantIncome = rng.normal(1500, 1000, size=n).clip(0)
    # LoanAmount roughly scales with incomes
    LoanAmount = ((ApplicantIncome * 0.12 + CoapplicantIncome * 0.05) / 10).round().clip(20)
    Loan_Amount_Term = rng.choice([360.0, 120.0, 180.0, 240.0, 300.0], size=n, p=[0.7,0.05,0.05,0.1,0.1])
    Credit_History = rng.choice([1.0, 0.0], size=n, p=[0.8, 0.2])
    Property_Area = rng.choice(['Urban','Rural','Semiurban'], size=n, p=[0.35,0.25,0.4])
    # Build approval probability with some dependence on credit history and income
    p = 0.15 + 0.55*Credit_History + (ApplicantIncome/20000.0) + (CoapplicantIncome/60000.0) - (LoanAmount/1000.0)
    p = np.clip(p, 0.01, 0.99)
    rng2 = rng.rand(n)
    Loan_Status = np.where(rng2 < p, 'Y', 'N')
    Loan_ID = [f"LP{1000+i}" for i in range(n)]
    df = pd.DataFrame({
        'Loan_ID': Loan_ID,
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome.round(0).astype(int),
        'CoapplicantIncome': CoapplicantIncome.round(0).astype(int),
        'LoanAmount': LoanAmount.astype(int),
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area,
        'Loan_Status': Loan_Status
    })
    df.to_csv(path, index=False)
    print(f"Synthetic dataset created and saved to {path} (rows={len(df)})")
    return df

def load_dataset(path='loan_data.csv'):
    if os.path.exists(path):
        print(f"Loading dataset from {path}")
        return pd.read_csv(path)
    else:
        print(f"{path} not found — creating a synthetic dataset.")
        return create_synthetic_dataset(path)

def build_preprocessor(numeric_features, categorical_features):
    # Numeric pipeline: median impute + standard scaling
    from sklearn.pipeline import Pipeline as SKPipeline
    num_pipeline = SKPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Categorical pipeline: mode impute + one-hot encoding
    cat_pipeline = SKPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    return preprocessor

def main():
    # 1) Load data
    df = load_dataset('loan_data.csv')
    print("\nSample rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    # 2) Basic cleaning: drop Loan_ID (identifier)
    X = df.drop(columns=['Loan_ID','Loan_Status'])
    y = df['Loan_Status'].map({'Y':1, 'N':0})  # encode target as 1/0

    # 3) Feature lists
    numeric_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # 4) Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RND, stratify=y
    )
    print(f"\nTrain rows: {len(X_train)}  Test rows: {len(X_test)}")

    # 5) Build preprocessor and pipelines
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipe_log = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=RND))
    ])
    pipe_tree = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=RND))
    ])

    # 6) Train both models
    print("\nTraining Logistic Regression...")
    pipe_log.fit(X_train, y_train)
    print("Training Decision Tree...")
    pipe_tree.fit(X_train, y_train)

    # 7) Evaluate both on test set
    models = [('Logistic Regression', pipe_log), ('Decision Tree', pipe_tree)]
    results = {}
    for name, pipe in models:
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n== {name} ==")
        print("Accuracy:", round(acc, 4))
        print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        results[name] = acc

    # 8) Select best model
    best_name = max(results, key=results.get)
    best_pipeline = pipe_log if best_name == 'Logistic Regression' else pipe_tree
    print(f"\nBest model by test accuracy: {best_name} (accuracy={results[best_name]:.4f})")

    # 9) Save the best pipeline to disk
    save_path = 'best_model.joblib'
    joblib.dump(best_pipeline, save_path)
    print(f"Saved best model pipeline to {save_path}")

    # 10) Demonstrate loading and predicting with saved model
    print("\nLoading saved model and making a sample prediction...")
    loaded = joblib.load(save_path)

    # Example single-sample input - change values to test different inputs
    sample = pd.DataFrame([{
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 6000,
        'CoapplicantIncome': 0,
        'LoanAmount': 120,
        'Loan_Amount_Term': 360.0,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }])
    print("Sample input:")
    print(sample.to_dict(orient='records')[0])
    pred = loaded.predict(sample)[0]
    pred_proba = loaded.predict_proba(sample)[0] if hasattr(loaded, "predict_proba") else None
    print("Predicted Loan_Status (1=Approved,0=Rejected):", int(pred))
    if pred_proba is not None:
        print("Prediction probabilities (class 0, class 1):", pred_proba.round(4))

    # 11) (Optional) Simple visualization of model accuracies
    try:
        names = list(results.keys())
        accs = [results[n] for n in names]
        plt.figure(figsize=(6,4))
        plt.bar(names, accs)
        plt.ylim(0,1)
        plt.title('Model test accuracies')
        plt.ylabel('Accuracy')
        for i,v in enumerate(accs):
            plt.text(i, v+0.01, f"{v:.3f}", ha='center')
        plt.tight_layout()
        plt.savefig('model_accuracies.png')
        print("\nSaved model accuracy bar chart to model_accuracies.png")
    except Exception as e:
        print("Could not create chart:", e)

if __name__ == '__main__':
    main()
