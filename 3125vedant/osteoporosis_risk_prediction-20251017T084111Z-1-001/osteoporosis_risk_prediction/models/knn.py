from sklearn.neighbors import KNeighborsClassifier
from utils.evaluation import evaluate_model

def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\n=== K-Nearest Neighbors ===")
    evaluate_model(y_test, y_pred, y_prob)
    return model
