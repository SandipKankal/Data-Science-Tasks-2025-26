from sklearn.svm import SVC
from utils.evaluation import evaluate_model

def train_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\n=== Support Vector Machine ===")
    evaluate_model(y_test, y_pred, y_prob)
    return model
