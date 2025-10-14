# -*- coding: utf-8 -*-
# Cirrhosis classification: clean pipeline + robust metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# 1) Load & quick inspect
# -----------------------------
df = pd.read_csv("cirrhosis.csv")

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(3))
print(df.isnull().sum().sort_values(ascending=False).head(10))

# -----------------------------
# 2) Define target & optional drops
# -----------------------------
TARGET = "Status"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found. Available: {df.columns.tolist()}")

# (Optional) Drop obvious identifier-only columns if present (keeps model honest)
maybe_id_cols = [c for c in ["ID", "id", "PatientID", "N_Days"] if c in df.columns]
X = df.drop([TARGET] + maybe_id_cols, axis=1)
y = df[TARGET]

# -----------------------------
# 3) Split (stratified to preserve class balance)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4) Preprocessing
#    - numerics: median impute + (optionally) scale
#    - categoricals: most_frequent impute + one-hot
# -----------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())  # scaling helps LR/SVM/KNN
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

# -----------------------------
# 5) Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", n_jobs=None
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", probability=True, random_state=42, class_weight="balanced"
    ),
    "KNN (k=11)": KNeighborsClassifier(n_neighbors=11, n_jobs=None if hasattr(KNeighborsClassifier(), "n_jobs") else None)
}

# -----------------------------
# 6) Train, predict, evaluate
# -----------------------------
def safe_roc_auc(y_true, y_proba, labels):
    """
    Works for binary or multiclass.
    y_proba: array of shape (n_samples, n_classes) or (n_samples,)
    """
    try:
        if y_proba.ndim == 1:  # binary with 1d probs
            return roc_auc_score(y_true, y_proba)
        else:
            # multiclass
            return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=labels)
    except Exception:
        return np.nan

results = []
conf_mats = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # Probabilities for AUC
    # Some classifiers expose predict_proba; if not, fall back to decision_function
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = pipe.predict_proba(X_test)
        if proba.shape[1] == 2:  # binary → take positive class
            y_proba_for_auc = proba[:, 1]
        else:
            y_proba_for_auc = proba  # multiclass matrix
    elif hasattr(pipe.named_steps["model"], "decision_function"):
        df_scores = pipe.decision_function(X_test)
        # Map decision_function to a 2D array for multiclass AUC handling
        y_proba_for_auc = df_scores if df_scores.ndim > 1 else 1 / (1 + np.exp(-df_scores))
    else:
        y_proba_for_auc = np.zeros(len(y_pred))  # will yield NaN AUC safely

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    labels_sorted = np.unique(y_train)
    auc = safe_roc_auc(y_test, y_proba_for_auc, labels_sorted)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision_w": p,
        "Recall_w": r,
        "F1_w": f1,
        "ROC_AUC_w/OVR": auc
    })

    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    conf_mats[name] = (cm, labels_sorted)

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\n=== Results (sorted by Accuracy) ===")
print(results_df.to_string(index=False))

# -----------------------------
# 7) Plots
# -----------------------------
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.title("Model Accuracy")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
melted = results_df.melt(id_vars="Model", value_vars=["Precision_w", "Recall_w", "F1_w"], var_name="Metric", value_name="Score")
sns.barplot(data=melted, x="Model", y="Score", hue="Metric")
plt.title("Precision / Recall / F1 (weighted)")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()

# Confusion matrices
for name, (cm, labels_sorted) in conf_mats.items():
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_sorted, yticklabels=labels_sorted)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 8) (Optional) Feature importances for RandomForest
# -----------------------------
rf_name = "Random Forest"
if rf_name in models:
    # Refit a dedicated pipeline to get feature names correctly
    rf_pipe = Pipeline(steps=[("prep", preprocessor), ("model", models[rf_name])])
    rf_pipe.fit(X_train, y_train)

    # Extract feature names from ColumnTransformer
    feature_names = []
    # Numeric names (unchanged by scaler)
    feature_names.extend(num_cols)
    # Categorical names (expanded by OneHot)
    if cat_cols:
        ohe = rf_pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        cat_expanded = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names = num_cols + cat_expanded

    importances = rf_pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(20)

    print("\nTop 20 features (Random Forest):")
    print(fi.to_string(index=False))

    plt.figure(figsize=(8, 6))
    sns.barplot(data=fi, x="importance", y="feature")
    plt.title("Random Forest – Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()
