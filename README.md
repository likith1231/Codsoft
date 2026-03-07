CodSoft Machine Learning Tasks
==============================

This repository contains three ML tasks for your CodSoft internship:

- **Task 1 – Credit Card Fraud Detection**
- **Task 2 – Customer Churn Prediction**
- **Task 3 – SMS Spam Detection**


Common setup
------------

1. Install Python 3.8+.
2. Open a terminal in the `Codsoft` folder.
3. (Optional but recommended) create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

On PowerShell you might need:

```powershell
.\.venv\Scripts\Activate.ps1
```

All three tasks share the same `requirements.txt`.


Task 1 – Credit Card Fraud Detection
------------------------------------

- **Goal**: Detect fraudulent credit card transactions using the Kaggle dataset  
  `https://www.kaggle.com/datasets/kartik2112/fraud-detection`.
- **Main file**: `fraud_detection.py`

Expected data files (place them in `Codsoft/data/`):

- `fraudTrain.csv`
- `fraudTest.csv` (optional, script mainly uses `fraudTrain.csv`)

Run:

```bash
python fraud_detection.py
```

What the script does:

- Loads `data/fraudTrain.csv`.
- Builds features and target (`is_fraud`).
- Uses a preprocessing + **Logistic Regression** pipeline (scaling + one-hot encoding + class weights).
- Prints class balance, confusion matrix, classification report, and ROC-AUC.
- Saves the trained model to `models/logreg_fraud_model.joblib`.


Task 2 – Customer Churn Prediction
----------------------------------

- **Goal**: Predict whether a bank customer will churn using historical data (demographics + account info).  
- **Dataset**: Kaggle bank customer churn dataset  
  `https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction`
- **Main file**: `churn_prediction.py`

Expected data file (place in `Codsoft/data/`):

- `Churn_Modelling.csv`

Run:

```bash
python churn_prediction.py
```

What the script does:

- Loads the churn dataset.
- Drops identifier columns like `RowNumber`, `CustomerId`, `Surname`.
- Uses customer features (e.g., `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, etc.).
- Builds a preprocessing pipeline (numeric scaling + one-hot encoding for categoricals).
- Trains and evaluates **three models**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Prints metrics (accuracy, precision, recall, F1, ROC-AUC) for each model.


Task 3 – SMS Spam Detection
---------------------------

- **Goal**: Classify SMS messages as **spam** or **ham** (legitimate).  
- **Dataset**: UCI SMS Spam Collection dataset on Kaggle  
  `https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset`
- **Main file**: `sms_spam_detection.py`

Expected data file (place in `Codsoft/data/`):

- `spam.csv`

Run:

```bash
python sms_spam_detection.py
```

What the script does:

- Loads the SMS dataset.
- Uses the message text as input and the spam/ham label as target.
- Applies **TF-IDF** text vectorization.
- Trains and evaluates several classifiers (Naive Bayes, Logistic Regression, Linear SVM).
- Prints accuracy, precision, recall, and F1-score for each model.


Notes
-----

- If your downloaded CSV file names differ, update the file paths at the top of each script.
- For your internship report, you can:
  - Add EDA plots (matplotlib / seaborn).
  - Tune model hyperparameters.
  - Compare models in tables and discuss trade-offs.

