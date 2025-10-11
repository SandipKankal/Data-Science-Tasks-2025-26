import numpy as np
import os
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from utils import save_model

def get_models():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC()),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier())
    }

def train_model(model_name, use_smote=False):
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("✅ SMOTE applied to balance dataset.")

    models = get_models()
    if model_name == "all":
        for name, model in models.items():
            print(f"🚀 Training {name}...")
            model.fit(X_train, y_train)
            save_model(model, f"models/model_{name}.joblib")
    else:
        model = models[model_name]
        print(f"🚀 Training {model_name}...")
        model.fit(X_train, y_train)
        save_model(model, f"models/model_{model_name}.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--use_smote", action="store_true")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    train_model(args.models, args.use_smote)
