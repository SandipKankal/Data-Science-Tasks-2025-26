Credit Card Fraud Detection 💳
This project uses machine learning to identify fraudulent credit card transactions. It tackles the core challenge of a highly imbalanced dataset, where fraudulent cases are extremely rare compared to normal transactions.

📝 Project Overview
The main goal is to build a reliable model that can accurately flag fraudulent activity to prevent financial loss.

The Challenge (Imbalanced Data): In real-world data, over 99% of transactions are legitimate. A naive model could achieve 99% accuracy by simply guessing "not fraud" every time, making it useless.

The Solution (SMOTE): To overcome this, the project uses the Synthetic Minority Over-sampling Technique (SMOTE). This technique intelligently generates new, synthetic examples of the minority class (fraud) in the training set. This balancing act ensures the model learns the patterns of fraud effectively without being biased towards the majority class.

Model Comparison: The project trains and compares two powerful algorithms:

A Decision Tree

A Neural Network

📊 Dataset
This project uses the "Credit Card Fraud Detection" dataset from Kaggle, which contains anonymized transaction data.

Source: Kaggle Credit Card Fraud Detection Dataset

The script requires the downloaded file to be named creditcard.csv and placed in the project's root directory.

⚙️ Requirements
You need Python 3 and the following libraries to run this project:

TensorFlow

scikit-learn

Imbalanced-learn (for SMOTE)

Pandas

You can install all required libraries by running this command:

Bash

pip install tensorflow scikit-learn imblearn pandas
🚀 How to Run
Ensure creditcard.csv and run_fraud_detection.py are in the same folder.

Open your terminal and navigate to the project folder.

Run the script with the following command:

Bash

python run_fraud_detection.py
📈 Results & Conclusion
The models were evaluated on their ability to detect the Fraud class. For this problem, Recall (the ability to find all actual fraud cases) and F1-Score (the balance between precision and recall) are the most important metrics.

Here are the results from the test run:

Model	Precision (Fraud)	Recall (Fraud)	F1-Score (Fraud)
Decision Tree	0.34	0.78	0.47
Neural Network	0.54	0.84	0.65

Export to Sheets
Conclusion: Neural Network is the Winner 🏆
The Neural Network is the superior model for this task. Here's why:

Higher Recall (84%): It successfully identified 84% of all fraudulent transactions in the test set, missing fewer cases than the Decision Tree (78%).

Higher Precision (54%): When it flagged a transaction as fraud, it was correct 54% of the time, resulting in fewer false alarms compared to the Decision Tree (34%).

Better Overall F1-Score (0.65): The significantly higher F1-Score shows that the Neural Network provides a much better balance of catching fraud without incorrectly flagging legitimate transactions.