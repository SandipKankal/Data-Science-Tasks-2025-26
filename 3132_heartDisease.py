# ==========================================================
# STEP 1: Import Libraries
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score

# Machine Learning Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# ==========================================================
# STEP 2: Load Dataset
# ==========================================================
df = pd.read_csv('heart_disease_dataset.csv')   # ensure correct file path
print("✅ Dataset loaded successfully!")
print(df.shape)
df.head()
# ==========================================================
# STEP 5: Handle Missing Values (Fix NaN / '?' Error)
# ==========================================================
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# 1️⃣ Replace '?' with NaN and ensure all features are numeric
X = X.replace('?', np.nan)
X = X.apply(pd.to_numeric, errors='coerce')

# 2️⃣ Split data again after cleaning
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Impute missing numeric values using median
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ✅ Check for NaNs after imputation
print("NaNs in train:", np.isnan(X_train).sum())
print("NaNs in test:", np.isnan(X_test).sum())
print("✅ Missing values handled successfully!")

# ==========================================================
# STEP 3: Explore Dataset
# ==========================================================
print(df.info())
print(df.isnull().sum())   # Check missing values
# ==========================================================
# STEP 4: Split Features & Target
# ==========================================================
X = df.drop('target', axis=1)
y = df['target']

# Split dataset into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ==========================================================
# STEP 6: Feature Scaling
# ==========================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ==========================================================
# STEP 7: Apply 6 Algorithms
# ==========================================================

# --- 1. K-Nearest Neighbors ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# --- 2. Linear Regression ---
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)
print("Linear Regression R² Score:", r2_score(y_test, y_pred_lr))

# --- 3. Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)# ==========================================================
# STEP 10: Gradient Boosting Classifier
# ==========================================================
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Initialize model
gb = GradientBoostingClassifier(random_state=42)

# Train model
gb.fit(X_train, y_train)

# Predict on test data
y_pred_gb = gb.predict(X_test)

# Evaluate performance
print("🌟 Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gb))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_gb)
plt.figure(figsize=(4, 3))
plt.title("Gradient Boosting - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.imshow(cm, cmap='Blues')
plt.colorbar()
for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, cm[i][j], ha='center', va='center', color='black')
plt.show()
# ==========================================================
# STEP 11: Confusion Matrices for All Algorithms
# ==========================================================
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create a dictionary of model predictions (excluding Linear Regression)
model_predictions = {
    "KNN": y_pred_knn,
    "Logistic Regression": y_pred_log,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm,
    "Gradient Boosting": y_pred_gb
}

# Plot confusion matrix for each model
for name, y_pred in model_predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

y_pred_log = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

# --- 4. Decision Tree ---
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# --- 5. Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# --- 6. Support Vector Machine (SVM) ---
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
