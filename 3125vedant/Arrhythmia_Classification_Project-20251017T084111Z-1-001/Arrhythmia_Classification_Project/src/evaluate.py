import numpy as np
import argparse
from sklearn.metrics import accuracy_score, classification_report
from utils import load_model, plot_confusion_matrix

def evaluate(model_name):
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    model = load_model(f"models/model_{model_name}.joblib")
    y_pred = model.predict(X_test)

    print(f"📊 Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.model)
