from sklearn.linear_model import LogisticRegression
from utils.evaluation import evaluate_model

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\n=== Logistic Regression ===")
    evaluate_model(y_test, y_pred, y_prob)
    return model
