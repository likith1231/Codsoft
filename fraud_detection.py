from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "fraudTrain.csv"

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "logreg_fraud_model.joblib"

# Set to None to use the full dataset (may be slow on some machines).
N_ROWS = 200_000


def load_train_data(n_rows: int | None = N_ROWS) -> pd.DataFrame:
    """Load the Kaggle fraud training data."""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Expected training file at '{TRAIN_FILE}'. "
            "Please download the Kaggle dataset and place 'fraudTrain.csv' in the 'data/' folder."
        )

    df = pd.read_csv(TRAIN_FILE, nrows=n_rows)
    return df


def build_features_and_target(df: pd.DataFrame):
    """Split raw dataframe into feature matrix X and target vector y."""
    target_col = "is_fraud"
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[target_col]

    # Drop columns that are identifiers, high-cardinality strings, or not useful for prediction
    # (adjust this list based on your EDA and internship requirements).
    drop_cols = [
        target_col,
        "trans_date_trans_time",
        "dob",
        "street",
        "city",
        "zip",
        "job",
        "last",
        "first",
        "trans_num",
        "cc_num",
    ]

    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols)

    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create a preprocessing + classification pipeline."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Logistic Regression with class_weight to address class imbalance
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_and_evaluate(
    n_rows: int | None = N_ROWS,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Train the model on the Kaggle dataset and print evaluation metrics."""
    df = load_train_data(n_rows=n_rows)
    print(f"Loaded {len(df):,} rows from '{TRAIN_FILE}'.")

    # Show class distribution
    if "is_fraud" in df.columns:
        class_counts = df["is_fraud"].value_counts()
        print("\nClass distribution (counts):")
        print(class_counts)
        print("\nClass distribution (fraction):")
        print((class_counts / class_counts.sum()).rename("fraction"))

    X, y = build_features_and_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline(X_train)

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification report:")
    print(classification_report(y_val, y_pred, digits=4))

    roc_auc = roc_auc_score(y_val, y_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH.resolve()}")


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()

