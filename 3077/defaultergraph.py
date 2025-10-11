import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# --- 1. Load the Data ---
# This script assumes 'breast-cancer.data' is in the same folder.
try:
    # Define the column names based on the 'breast-cancer.names' file
    column_names = [
        'class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
        'deg-malig', 'breast', 'breast-quad', 'irradiat'
    ]
    
    # Load the dataset, correctly identifying '?' as a missing value
    df = pd.read_csv('breast-cancer.data', header=None, names=column_names, na_values='?')
    print("✅ Data loaded successfully!")

except FileNotFoundError:
    print("❌ Error: 'breast-cancer.data' not found.")
    print("Please make sure the data file is in the same directory as this script.")
    # Exit the script if the file isn't found
    exit()


# --- 2. Generate and Save Charts ---

# Chart 1: Distribution of Recurrence Events
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=df, order=df['class'].value_counts().index)
plt.title('Distribution of Recurrence Events', fontsize=16)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.tight_layout()
plt.savefig('1_class_distribution.png')
print("✅ Saved '1_class_distribution.png'")

# Chart 2: Age Distribution by Recurrence Class
plt.figure(figsize=(12, 7))
age_order = sorted(df['age'].dropna().unique())
sns.countplot(x='age', hue='class', data=df, order=age_order)
plt.title('Age Distribution by Recurrence Class', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.legend(title='Recurrence Event')
plt.tight_layout()
plt.savefig('2_age_distribution_by_class.png')
print("✅ Saved '2_age_distribution_by_class.png'")

# Chart 3: Tumor Size vs. Recurrence
plt.figure(figsize=(14, 8))
tumor_size_order = sorted(df['tumor-size'].dropna().unique(), key=lambda x: int(x.split('-')[0]))
sns.countplot(x='tumor-size', hue='class', data=df, order=tumor_size_order)
plt.title('Tumor Size vs. Recurrence', fontsize=16)
plt.xlabel('Tumor Size (mm)', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Recurrence Event')
plt.tight_layout()
plt.savefig('3_tumor_size_vs_recurrence.png')
print("✅ Saved '3_tumor_size_vs_recurrence.png'")

# Chart 4: Feature Importance (Requires model training)
print("⚙️  Preprocessing data and training a model for feature importance...")
# First, preprocess the data
X = df.drop('class', axis=1)
y = df['class']

# Handle missing values by replacing them with the most frequent value in their column
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Convert all text categories to numbers so the model can process them
encoder = OrdinalEncoder()
X_encoded = pd.DataFrame(encoder.fit_transform(X_imputed), columns=X.columns)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_encoded, y)

# Get and sort the feature importances from the trained model
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance for Predicting Recurrence', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('4_feature_importance.png')
print("✅ Saved '4_feature_importance.png'")
print("\nAll charts have been generated and saved!")