import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("churn.csv")

# remove useless columns
df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)

# convert gender
df["Gender"] = df["Gender"].map({"Male":1, "Female":0})

# convert country
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model.pkl","wb"))

print("✅ Model trained successfully")
