from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("data")
CHURN_FILE = DATA_DIR / "Churn_Modelling.csv"


def load_churn_data() -> pd.DataFrame:
    """Load the bank customer churn dataset."""
    if not CHURN_FILE.exists():
        raise FileNotFoundError(
            f"Expected churn dataset at '{CHURN_FILE}'. "
            "Please download it from Kaggle and place 'Churn_Modelling.csv' in the 'data/' folder."
        )

    df = pd.read_csv(CHURN_FILE)
    return df


def build_features_and_target(df: pd.DataFrame):
    """Prepare X (features) and y (target) for churn prediction."""
    target_col = "Exited"  # 1 = churned, 0 = not churned
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[target_col]

    # Drop identifier / non-predictive columns
    drop_cols = [
        target_col,
        "RowNumber",
        "CustomerId",
        "Surname",
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create the preprocessing transformer for numeric and categorical features."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def evaluate_models(X_train, X_test, y_train, y_test) -> None:
    """Train and evaluate multiple models for churn prediction."""
    preprocessor = build_preprocessor(X_train)

    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    for name, clf in models.items():
        print("=" * 80)
        print(f"Training model: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", clf),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # ROC-AUC can use predicted probabilities when available
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            # fall back to decision function (e.g., linear SVM-like models)
            if hasattr(pipeline.named_steps["model"], "decision_function"):
                y_proba = pipeline.decision_function(X_test)
            else:
                y_proba = None

        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc:.4f}")

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, digits=4))

        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"ROC-AUC: {roc_auc:.4f}")
        else:
            print("ROC-AUC: not available for this model (no probability or decision scores).")


def train_and_evaluate_churn() -> None:
    """End-to-end training and evaluation for the churn task."""
    df = load_churn_data()
    print(f"Loaded churn dataset with {len(df):,} rows and {len(df.columns)} columns.")

    if "Exited" in df.columns:
        print("\nChurn distribution (Exited):")
        counts = df["Exited"].value_counts()
        print(counts)
        print("\nChurn distribution (fraction):")
        print((counts / counts.sum()).rename("fraction"))

    X, y = build_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    evaluate_models(X_train, X_test, y_train, y_test)


def main() -> None:
    train_and_evaluate_churn()


if __name__ == "__main__":
    main()

