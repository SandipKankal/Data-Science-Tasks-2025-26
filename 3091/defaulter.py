import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load dataset
# Added encoding='latin-1' to handle potential encoding issues
df = pd.read_csv('/content/Liver Patient Dataset (LPD)_train.csv', encoding='latin-1')

# Quick data info
print("Dataset shape:", df.shape)
print("Target distribution:")
print(df['Result'].value_counts())


# Handle missing values - simple approach
df = df.dropna()

# Convert target to binary (assuming 1=patient, 2=healthy)
df['Result'] = df['Result'].map({1: 1, 2: 0})

# Convert categorical columns if any
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Result':
        df[col] = df[col].astype('category').cat.codes

# Split data
X = df.drop('Result', axis=1)
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Initialize models with basic parameters
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate quickly
results = {}
for name, model in models.items():
    print(f"Training {name}...")

    # Use scaled data for specific models
    if name in ['Logistic Regression', 'SVM', 'KNN']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }

    print(f"{name} Accuracy: {accuracy:.4f}")
    print("-" * 40)



    # Simple results comparison
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{name:20} : {result['accuracy']:.4f}")

# Best model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest Model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")


# Detailed analysis only for best model
best_model_name = best_model[0]
best_predictions = best_model[1]['predictions']

print(f"\nDetailed Report for {best_model_name}:")
print(classification_report(y_test, best_predictions))
