"""
Setup script: downloads/prepares datasets for all 3 Codsoft ML tasks.
Run once: python setup_datasets.py
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def prepare_sms_spam() -> None:
    """Convert UCI SMSSpamCollection to spam.csv (v1=label, v2=message)."""
    src = DATA_DIR / "SMSSpamCollection"
    dst = DATA_DIR / "spam.csv"
    if not src.exists():
        print(f"Skipping SMS: {src} not found (extract smsspam.zip first)")
        return

    rows = []
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                rows.append({"v1": parts[0], "v2": parts[1]})
            else:
                rows.append({"v1": "ham", "v2": line})

    df = pd.DataFrame(rows)
    df.to_csv(dst, index=False)
    print(f"Created {dst} with {len(df):,} rows")


def create_fraud_train() -> None:
    """Create synthetic fraud dataset matching fraudTrain.csv schema."""
    import numpy as np

    np.random.seed(42)
    n = 50_000  # smaller for quick runs

    categories = ["grocery_pos", "gas_transport", "home", "shopping_net", "entertainment", "food_dining"]
    states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA"]

    df = pd.DataFrame({
        "trans_date_trans_time": pd.date_range("2020-01-01", periods=n, freq="min").astype(str),
        "cc_num": np.random.randint(100000, 999999, n).astype(object),  # simplified for synthetic
        "merchant": [f"merch_{i % 1000}" for i in range(n)],
        "category": np.random.choice(categories, n),
        "amt": np.clip(np.random.lognormal(2, 1.5, n), 1, 500),
        "first": [f"First{i % 500}" for i in range(n)],
        "last": [f"Last{i % 500}" for i in range(n)],
        "gender": np.random.choice(["M", "F"], n),
        "street": [f"Street {i}" for i in range(n)],
        "city": [f"City{i % 100}" for i in range(n)],
        "state": np.random.choice(states, n),
        "zip": np.random.randint(10000, 99999, n),
        "lat": np.random.uniform(25, 45, n),
        "long": np.random.uniform(-120, -75, n),
        "city_pop": np.random.randint(1000, 500000, n),
        "job": [f"Job{i % 50}" for i in range(n)],
        "dob": pd.date_range("1970-01-01", periods=n, freq="D").astype(str),
        "trans_num": [f"trans_{i}" for i in range(n)],
    })

    # Fraud: ~1.5% imbalanced
    fraud_prob = 0.015
    df["is_fraud"] = (np.random.random(n) < fraud_prob).astype(int)
    # Slight signal: higher amounts and certain categories more likely fraud
    fraud_idx = df["is_fraud"] == 1
    df.loc[fraud_idx, "amt"] *= np.random.uniform(1.2, 2.0, fraud_idx.sum())
    df.loc[fraud_idx, "category"] = np.random.choice(["shopping_net", "entertainment", "food_dining"], fraud_idx.sum())

    out = DATA_DIR / "fraudTrain.csv"
    df.to_csv(out, index=False)
    print(f"Created {out} with {len(df):,} rows (fraud rate: {df['is_fraud'].mean():.2%})")


def create_churn_data() -> None:
    """Create synthetic churn dataset matching Churn_Modelling.csv schema."""
    import numpy as np

    np.random.seed(42)
    n = 10_000

    df = pd.DataFrame({
        "RowNumber": range(1, n + 1),
        "CustomerId": np.random.randint(100000, 999999, n),
        "Surname": [f"Surname{i}" for i in range(n)],
        "CreditScore": np.clip(np.random.normal(650, 80, n), 350, 850).astype(int),
        "Geography": np.random.choice(["France", "Germany", "Spain"], n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Age": np.clip(np.random.normal(38, 10, n), 18, 92).astype(int),
        "Tenure": np.random.randint(0, 11, n),
        "Balance": np.random.uniform(0, 250000, n),
        "NumOfProducts": np.random.choice([1, 2, 3, 4], n),
        "HasCrCard": np.random.choice([0, 1], n),
        "IsActiveMember": np.random.choice([0, 1], n),
        "EstimatedSalary": np.random.uniform(10000, 200000, n),
    })

    # Churn ~20% with some signal from Balance, Age, NumOfProducts
    logit = -2 + 0.00001 * df["Balance"] + 0.02 * df["Age"] - 0.3 * df["NumOfProducts"]
    prob = 1 / (1 + np.exp(-logit + np.random.normal(0, 0.5, n)))
    df["Exited"] = (np.random.random(n) < prob).astype(int)

    out = DATA_DIR / "Churn_Modelling.csv"
    df.to_csv(out, index=False)
    print(f"Created {out} with {len(df):,} rows (churn rate: {df['Exited'].mean():.2%})")


def main() -> None:
    print("Preparing datasets for Codsoft ML tasks...\n")

    prepare_sms_spam()
    create_fraud_train()
    create_churn_data()

    print("\nDone. You can now run:")
    print("  python fraud_detection.py")
    print("  python churn_prediction.py")
    print("  python sms_spam_detection.py")


if __name__ == "__main__":
    main()
