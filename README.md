CodSoft Machine Learning Internship Projects
============================================

This repository contains three machine learning projects developed as part of the CodSoft internship program.
Each project focuses on solving a real-world classification problem using appropriate machine learning algorithms and data preprocessing techniques.

Due to GitHub storage limitations, large dataset files `(.csv)` and generated model/database files `(.pkl, .db)` are not included in the repository.
They must be stored locally inside their respective task folders for the applications to run successfully.
```
Codsoft/
│
├── Task1_FraudDetection/
│   ├── app.py
│   ├── train_model.py
│   ├── model.pkl                 (generated after training)
│   ├── fraudTrain.csv            (local dataset)
│   ├── fraudTest.csv             (optional local dataset)
│   ├── templates/
│   └── static/
│
├── Task2_ChurnPrediction/
│   ├── app.py
│   ├── train_model.py
│   ├── model.pkl                 (generated after training)
│   ├── database.db               (generated automatically)
│   ├── Churn_Modelling.csv       (local dataset)
│   ├── templates/
│   └── static/
│
├── Task3_SpamDetection/
│   ├── app.py
│   ├── train_model.py
│   ├── spam_model.pkl            (generated)
│   ├── vectorizer.pkl            (generated)
│   ├── spam.db                   (generated)
│   ├── spam.csv                  (local dataset)
│   ├── templates/
│   └── static/
│
├── requirements.txt
└── README.md
```

Task 1 — Credit Card Fraud Detection
------------------------------------
- Develop a machine learning model to classify credit card transactions as fraudulent or legitimate using historical transaction data.
- Algorithms Implemented
- Logistic Regression
- Decision Tree
- Random Forest
- Dataset
- Credit Card Fraud Detection Dataset (Kaggle)
  
Dataset Source

- Kaggle Fraud Detection Dataset = 
https://www.kaggle.com/datasets/kartik2112/fraud-detection


Steps to Run
```
cd Task1_FraudDetection
python train_model.py
python app.py
```
After training, the file `model.pkl` will be generated automatically.


Task 2 — Customer Churn Prediction
----------------------------------
- Predict whether a customer will discontinue a subscription-based banking service using demographic and account usage features.
- Algorithms Implemented
- Logistic Regression
- Random Forest
- Gradient Boosting
- Dataset
- Bank Customer Churn Prediction Dataset (Kaggle)

Dataset Source

- Kaggle Bank Customer Churn Dataset = https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

Steps to Run
```
cd Task2_ChurnPrediction
python train_model.py
python app.py
```
This process generates:

- `model.pkl`
- `database.db` (stores prediction history)

Task 3 — SMS Spam Detection
---------------------------

- Classify SMS messages as spam or legitimate using natural language processing techniques
- Techniques and Algorithms
- TF-IDF Vectorization
- Naive Bayes
- Logistic Regression
- Support Vector Machine
- Dataset
- SMS Spam Collection Dataset (Kaggle / UCI)

Dataset Source

- SMS Spam Collection Dataset = https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Steps to Run
```
cd Task3_SpamDetection
python train_model.py
python app.py
```
This process generates:

- `spam_model.pkl`
- `vectorizer.pkl`
- `spam.db`

Installation
------------
Install required dependencies:
- `pip install -r requirements.txt`

(Optional virtual environment setup)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Important Notes
---------------

- Keep all dataset files inside their respective task directories.
- Generated model and database files should remain local and is not pushed to GitHub.
- Each task is implemented as an independent machine learning pipeline with its own user interface and training workflow.
