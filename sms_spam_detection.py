from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_DIR = Path("data")
SMS_FILE = DATA_DIR / "spam.csv"


def load_sms_data() -> pd.DataFrame:
    """Load the SMS spam dataset."""
    if not SMS_FILE.exists():
        raise FileNotFoundError(
            f"Expected SMS dataset at '{SMS_FILE}'. "
            "Please download it from Kaggle and place 'spam.csv' in the 'data/' folder."
        )

    # The UCI SMS dataset usually comes with extra unnamed columns; we'll drop them.
    df = pd.read_csv(SMS_FILE, encoding="latin-1")

    # Common structure: columns v1 (label), v2 (message), plus some unnamed extras.
    if "v1" not in df.columns or "v2" not in df.columns:
        raise KeyError(
            "Expected columns 'v1' (label) and 'v2' (message text) not found in dataset.\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    return df


def prepare_data(df: pd.DataFrame):
    """Prepare train/test split for SMS spam detection."""
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def evaluate_sms_models(X_train, X_test, y_train, y_test) -> None:
    """Train and evaluate several classifiers for SMS spam detection."""
    base_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20_000,
    )

    models: dict[str, object] = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    for name, clf in models.items():
        print("=" * 80)
        print(f"Training model: {name}")

        pipeline = Pipeline(
            steps=[
                ("tfidf", base_vectorizer),
                ("model", clf),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, digits=4))


def train_and_evaluate_sms() -> None:
    """End-to-end training and evaluation for the SMS spam task."""
    df = load_sms_data()
    print(f"Loaded SMS dataset with {len(df):,} messages.")

    print("\nLabel distribution (ham/spam):")
    counts = df["label"].value_counts()
    print(counts)
    print("\nLabel distribution (fraction):")
    print((counts / counts.sum()).rename("fraction"))

    X_train, X_test, y_train, y_test = prepare_data(df)
    evaluate_sms_models(X_train, X_test, y_train, y_test)


def main() -> None:
    train_and_evaluate_sms()


if __name__ == "__main__":
    main()

