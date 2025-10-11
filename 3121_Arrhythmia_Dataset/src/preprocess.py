import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from utils import mean_impute

def preprocess(binary=True):
    data_path = "data/arrhythmia.data"
    df = pd.read_csv(data_path, header=None, na_values='?')
    print("✅ Dataset loaded successfully!")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Binary classification (Normal vs Abnormal)
    if binary:
        y = np.where(y == 1, 0, 1)

    X = mean_impute(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    os.makedirs("data", exist_ok=True)
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)
    print("✅ Preprocessing completed and data saved in 'data/' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=bool, default=True)
    args = parser.parse_args()
    preprocess(binary=args.binary)
