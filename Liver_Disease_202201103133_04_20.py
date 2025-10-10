import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("/content/Liver Patient Dataset (LPD)_train.csv", encoding='latin1')

# Clean column names (remove leading/trailing whitespaces)
df.columns = df.columns.str.strip()


le = LabelEncoder()
df['Gender of the patient'] = le.fit_transform(df['Gender of the patient'])


df = df.dropna()


X = df.drop('Result', axis=1)
y = df['Result']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



#Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# Train model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predictions
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

# Evaluation
print("=" * 60)
print(f"Model: {dt.__class__.__name__}")
print("Parameters:", dt.get_params())
print("=" * 60)
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
print("=" * 60)
print("Classification Report:")
print(classification_report(y_test, y_pred_test))
print("=" * 60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


#Random Forest 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Evaluation
print("=" * 60)
print(f"Model: {rf.__class__.__name__}")
print("Parameters:", rf.get_params())
print("=" * 60)
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
print("=" * 60)
print("Classification Report:")
print(classification_report(y_test, y_pred_test))
print("=" * 60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


#LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# Train model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Predictions
y_pred_train = lda.predict(X_train)
y_pred_test = lda.predict(X_test)

# Evaluation
print("=" * 60)
print(f"Model: {lda.__class__.__name__}")
print("Parameters:", lda.get_params())
print("=" * 60)
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
print("=" * 60)
print("Classification Report:")
print(classification_report(y_test, y_pred_test))
print("=" * 60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Linear Discriminant Analysis - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


#logistic Regression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# Train model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Predictions
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

# Evaluation
print("=" * 60)
print(f"Model: {lr.__class__.__name__}")
print("Parameters:", lr.get_params())
print("=" * 60)
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
print("=" * 60)
print("Classification Report:")
print(classification_report(y_test, y_pred_test))
print("=" * 60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


#KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

# Evaluation
print("=" * 60)
print(f"Model: {knn.__class__.__name__}")
print("Parameters:", knn.get_params())
print("=" * 60)
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
print("=" * 60)
print("Classification Report:")
print(classification_report(y_test, y_pred_test))
print("=" * 60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title("K-Nearest Neighbors - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


#SVM

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# Train model
svm = SVC()
svm.fit(X_train, y_train)

# Predictions
y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)

# Evaluation
print("=" * 60)
print(f"Model: {svm.__class__.__name__}")
print("Parameters:", svm.get_params())
print("=" * 60)
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
print("=" * 60)
print("Classification Report:")
print(classification_report(y_test, y_pred_test))
print("=" * 60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Support Vector Machine - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


#NN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
model.add(Dense(32, activation='relu'))                               # Second hidden layer
model.add(Dense(1, activation='sigmoid'))                             # Output layer

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Predictions
y_pred_train = (model.predict(X_train) > 0.5).astype("int32")
y_pred_test  = (model.predict(X_test) > 0.5).astype("int32")

# Evaluation
print("=" * 60)
print("Model: Neural Network")
print("Architecture: 64 -> 32 -> 1")
 
print("=" * 60)
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
print("=" * 60)
print("Classification Report:")
print(classification_report(y_test, y_pred_test))
print("=" * 60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title("Neural Network - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
