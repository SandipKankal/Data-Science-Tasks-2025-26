from sklearn.ensemble import RandomForestClassifier
from utils.evaluation import evaluate_model

def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\n=== Random Forest ===")
    evaluate_model(y_test, y_pred, y_prob)
    return model
