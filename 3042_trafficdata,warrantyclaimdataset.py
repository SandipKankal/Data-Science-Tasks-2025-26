

!pip install xgboost --quiet

from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('synthetic_traffic_data.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
df['is_weekend'] = df['dayofweek'] >= 5

df['total_vehicles'] = df[['cars', 'bikes', 'buses', 'trucks']].sum(axis=1)

df = df.drop(columns=['timestamp'])

X = df.drop(columns=['total_vehicles'])
y = df['total_vehicles']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

plt.figure(figsize=(12,6))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title('Actual vs Predicted Total Vehicle Counts')
plt.xlabel('Sample')
plt.ylabel('Vehicle Count')
plt.legend()
plt.show()





!pip install xgboost --quiet

from google.colab import files
uploaded = files.upload()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('warranty_claims_data.csv')

df['purchase_date'] = pd.to_datetime(df['purchase_date'])

reference_date = pd.Timestamp('2023-01-01')
df['days_since_purchase'] = (reference_date - df['purchase_date']).dt.days

df.drop(columns=['purchase_date'], inplace=True)

categorical_cols = ['region', 'product_category', 'claim_reason']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['is_authentic'])
y = df['is_authentic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ... (your earlier steps remain same until train-test split)

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.4f}")

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('XGBoost Confusion Matrix')
plt.show()


# Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))

cm_rf = confusion_matrix(y_test, rf_y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Step 1: Install necessary packages (if not already installed)
!pip install xgboost --quiet

# Step 2: Upload your CSV file
from google.colab import files
uploaded = files.upload()

# Step 3: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Step 4: Load the dataset
df = pd.read_csv('synthetic_traffic_data.csv')

# Step 5: Preprocess data and feature engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
df['is_weekend'] = df['dayofweek'] >= 5

# Calculate total vehicles as target variable
df['total_vehicles'] = df[['cars', 'bikes', 'buses', 'trucks']].sum(axis=1)

# Drop timestamp (not needed as a feature)
df = df.drop(columns=['timestamp'])

# Step 6: Define features (X) and target (y)
X = df.drop(columns=['total_vehicles'])
y = df['total_vehicles']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Step 9: Predict on test set with XGBoost
y_pred = xgb_model.predict(X_test)

# Step 10: Evaluate XGBoost model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"XGBoost MAE: {mae:.2f}")
print(f"XGBoost RMSE: {rmse:.2f}")
print(f"XGBoost R2 Score: {r2:.4f}")

# Step 11: Train Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Step 12: Predict on test set with Random Forest
rf_y_pred = rf_model.predict(X_test)

# Step 13: Evaluate Random Forest model
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_r2 = r2_score(y_test, rf_y_pred)

print(f"Random Forest MAE: {rf_mae:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"Random Forest R2 Score: {rf_r2:.4f}")

# Step 14: Plot actual vs predicted traffic for the first 100 samples in test set (XGBoost)
plt.figure(figsize=(12,6))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(y_pred[:100], label='XGBoost Predicted')
plt.plot(rf_y_pred[:100], label='Random Forest Predicted', alpha=0.7)
plt.title('Actual vs Predicted Total Vehicle Counts')
plt.xlabel('Sample')
plt.ylabel('Vehicle Count')
plt.legend()
plt.show()