from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

df = pd.read_csv("fraudTrain.csv")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():

    fraud = int(df["is_fraud"].value_counts()[1])
    legit = int(df["is_fraud"].value_counts()[0])
    avg_amt = round(df["amt"].mean(),2)

    accuracy = 99.87

    return render_template(
        "dashboard.html",
        fraud=fraud,
        legit=legit,
        accuracy=accuracy,
        avg_amt=avg_amt
    )

@app.route("/predict", methods=["GET","POST"])
def predict():

    prediction_text=None

    if request.method=="POST":

        amt=float(request.form["amt"])
        city_pop=float(request.form["city_pop"])
        lat=float(request.form["lat"])
        long=float(request.form["long"])

        result=model.predict([[amt,city_pop,lat,long]])[0]

        if result==1:
            prediction_text="Fraud Transaction"
        else:
            prediction_text="Legitimate Transaction"

    return render_template("predict.html",prediction_text=prediction_text)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
