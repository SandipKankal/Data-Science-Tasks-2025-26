from utils.preprocessing import load_and_preprocess_data
from models.logistic_regression import train_logistic_regression
from models.decision_tree import train_decision_tree
from models.random_forest import train_random_forest
from models.svm import train_svm
from models.knn import train_knn

def main():
    print("🔹 Loading and preprocessing data...")
    data_path = "data/patients_data.csv"

    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    print("\n🚀 Training models...\n")
    train_logistic_regression(X_train, X_test, y_train, y_test)
    train_decision_tree(X_train, X_test, y_train, y_test)
    train_random_forest(X_train, X_test, y_train, y_test)
    train_svm(X_train, X_test, y_train, y_test)
    train_knn(X_train, X_test, y_train, y_test)

    print("\n✅ All models trained and evaluated successfully!")

if __name__ == "__main__":
    main()
