# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings

# Ignore minor warnings
warnings.filterwarnings('ignore')

print("--- Credit Card Fraud Detection Model Execution Started ---")

# Step 2: Load the dataset
try:
    df = pd.read_csv('creditcard.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'creditcard.csv' not found!")
    print("Please download the dataset from Kaggle and place it in the same folder.")
    exit()

# Step 3: Data Preprocessing
print("⏳ Starting data preprocessing...")
# Scale 'Time' and 'Amount' columns, as they have a different range than the others
scaler = StandardScaler()
df['scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1) # Drop original columns
print("✅ Data preprocessing complete.")


# Step 4: Split data into features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class'] # 'Class' is the target: 1 for fraud, 0 for normal

# Split data, ensuring the class distribution is the same in train and test sets (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Data split. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


# Step 5: Handle Imbalanced Data using SMOTE
print("⏳ Handling class imbalance with SMOTE...")
# SMOTE creates synthetic samples for the minority class (fraud)
# IMPORTANT: Apply SMOTE only on the training data to avoid data leakage
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("✅ SMOTE applied. Training set is now balanced.")


# Step 6: Train the Machine Learning Models

# --- Decision Tree ---
print("⏳ Training Decision Tree model...")
dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train_resampled, y_train_resampled)
print("✅ Decision Tree model trained.")


# --- Neural Network ---
print("⏳ Building and training Neural Network model...")
nn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, input_dim=X_train_resampled.shape[1], activation='relu'), # Input layer
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model on the balanced data (epochs=5 for a quick run)
nn_model.fit(X_train_resampled, y_train_resampled, epochs=5, batch_size=256, verbose=1)
print("✅ Neural Network model trained.")


# Step 7: Evaluate the Models on the UNSEEN Test Data
print("\n--- Model Evaluation ---")
print("Evaluating models on the original, imbalanced test set...")

# Decision Tree Evaluation
dt_pred = dec_tree.predict(X_test)
print("\n--- Decision Tree Report ---")
print(classification_report(y_test, dt_pred, target_names=['Normal', 'Fraud']))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))

# Neural Network Evaluation
nn_pred_prob = nn_model.predict(X_test)
nn_pred = (nn_pred_prob > 0.5).astype(int) # Convert probabilities to 0 or 1
print("\n--- Neural Network Report ---")
print(classification_report(y_test, nn_pred, target_names=['Normal', 'Fraud']))
print("Confusion Matrix:\n", confusion_matrix(y_test, nn_pred))


# Step 8: Final Conclusion
print("\n--- Final Conclusion ---")
print("For fraud detection, accuracy is not the best metric.")
print("Focus on the 'Recall' and 'f1-score' for the 'Fraud' class.")
print("A higher 'Recall' means the model is better at catching actual fraud cases.")
print("The model with the better performance on the 'Fraud' class is the winner.")
print("\n--- Execution Finished ---")