import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(path):
    # Load dataset
    df = pd.read_csv(path)

    # Print column names for debugging
    print("🧩 Columns in your dataset:", list(df.columns))

    # Automatically detect the target column (label)
    possible_targets = [c for c in df.columns if 'osteoporosis' in c.lower() or 'label' in c.lower() or 'target' in c.lower()]
    if len(possible_targets) == 0:
        raise ValueError("❌ Could not find a target column (e.g., 'osteoporosis_label'). Please check your CSV file.")
    target_col = possible_targets[0]
    print(f"✅ Target column detected: {target_col}")

    # Fill missing numeric values with median
    df = df.fillna(df.median(numeric_only=True))

    # Encode categorical columns
    label_cols = df.select_dtypes(include=['object']).columns
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Split features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
