import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# --- 1. Load the Data ---
# Make sure 'breast-cancer.data' is in the same folder as your script
try:
    # Define the column names based on 'breast-cancer.names' file
    column_names = [
        'class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
        'deg-malig', 'breast', 'breast-quad', 'irradiat'
    ]
    
    # Load the dataset, replacing '?' with NaN for missing values
    df = pd.read_csv('breast-cancer.data', header=None, names=column_names, na_values='?')
    print("Data loaded successfully!")

except FileNotFoundError:
    print("Error: 'breast-cancer.data' not found.")
    print("Please make sure the data file is in the same directory as this script.")
    # Exit if the file isn't found
    exit()


# --- 2. Generate Charts ---

# Chart 1: Distribution of Recurrence Events
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=df, order=df['class'].value_counts().index)
plt.title('Distribution of Recurrence Events', fontsize=16)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.tight_layout()
plt.savefig('class_distribution.png')
print("Saved 'class_distribution.png'")

# Chart 2: Age Distribution by Recurrence Class
plt.figure(figsize=(12, 7))
age_order = sorted(df['age'].dropna().unique())
sns.countplot(x='age', hue='class', data=df, order=age_order)
plt.title('Age Distribution by Recurrence Class', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.legend(title='Recurrence Event')
plt.tight_layout()
plt.savefig('age_distribution_by_class.png')
print("Saved 'age_distribution_by_class.png'")

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
plt.savefig('tumor_size_vs_recurrence.png')
print("Saved 'tumor_size_vs_recurrence.png'")


# Chart 4: Feature Importance (Requires model training)
# First, we preprocess the data
X = df.drop('class', axis=1)
y = df['class']
# Handle missing values by replacing them with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
# Convert all text categories to numbers
encoder = OrdinalEncoder()
X_encoded = pd.DataFrame(encoder.fit_transform(X_imputed), columns=X.columns)
# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_encoded, y)
# Get and sort feature importances
importances = model.feature_import