import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# load dataset
df = pd.read_csv("fraudTrain.csv")

# select simple features
df = df[['amt','city_pop','lat','long','is_fraud']]

# remove missing values
df = df.dropna()

# features
X = df[['amt','city_pop','lat','long']]
y = df['is_fraud']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")
