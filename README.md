Credit Card Fraud Detection (Kaggle)
====================================

This project builds a **credit card fraud detection** model using the Kaggle dataset:
`https://www.kaggle.com/datasets/kartik2112/fraud-detection`.

It loads the transaction data, preprocesses features, trains a machine learning model,
and evaluates performance using metrics suited for imbalanced classification.


Project structure
-----------------

- `README.md` – project documentation and setup instructions  
- `requirements.txt` – Python dependencies  
- `fraud_detection.py` – main training & evaluation script  
- `data/` – folder where you place the downloaded Kaggle CSV files  
- `models/` – folder where trained models will be saved (created automatically)


1. Prerequisites
----------------

- Python 3.8+ installed
- `pip` available in your PATH

Optional but recommended:

- A virtual environment (so dependencies for this project stay isolated)


2. Setup environment
--------------------

Open a terminal in the project folder and run:

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


3. Download the dataset from Kaggle
-----------------------------------

1. Open the dataset page in your browser:  
   `https://www.kaggle.com/datasets/kartik2112/fraud-detection`
2. Sign in to Kaggle (account required).
3. Click **Download** to get the dataset as a `.zip` file.
4. Extract the archive locally.
5. Copy the CSV file(s), typically:
   - `fraudTrain.csv`
   - `fraudTest.csv`

   into a `data/` folder in this project (create it if it does not exist), so you have:

   - `data/fraudTrain.csv`
   - `data/fraudTest.csv`

If your CSV file names differ, update the paths inside `fraud_detection.py`.


4. Run the training script
--------------------------

From the project root, with the virtual environment activated:

```bash
python fraud_detection.py
```

The script will:

1. Load `data/fraudTrain.csv` (or fail with a clear error if it is missing).
2. Prepare features and target (`is_fraud`).
3. Split data into training and validation sets.
4. Build a preprocessing + logistic regression pipeline:
   - Scale numeric features.
   - One-hot encode categorical features.
   - Use class weighting to reduce class imbalance issues.
5. Train the model.
6. Print:
   - Class balance in the dataset.
   - Confusion matrix.
   - Precision, recall, F1-score.
   - ROC-AUC score.
7. Save the trained pipeline to `models/logreg_fraud_model.joblib`.


5. Modifying the script for your internship report
--------------------------------------------------

For your internship, you can extend or customize:

- **Exploratory data analysis (EDA)**:
  - Plot histograms of transaction amounts.
  - Plot class distribution (fraud vs non-fraud).
  - Analyse fraud rate by category, amount, time, etc.

- **Model experiments**:
  - Try other models like `RandomForestClassifier`, `XGBClassifier` (XGBoost), etc.
  - Adjust sampling size or use techniques such as SMOTE (from `imbalanced-learn`).

- **Reporting**:
  - Summarize your approach, preprocessing decisions, model choice, and metrics.
  - Compare multiple models in a small table (precision, recall, F1, ROC-AUC).

You can also move this logic into Jupyter notebooks if you prefer to present it
step by step (`pip install jupyter` and then `jupyter notebook`).

