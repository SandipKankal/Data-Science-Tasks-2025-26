import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"✅ Model saved as {filename}")

def load_model(filename):
    return joblib.load(filename)

def mean_impute(X):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(X)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
