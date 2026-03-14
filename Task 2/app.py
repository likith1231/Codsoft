from flask import Flask, render_template, request
import pickle
import sqlite3
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

# create database
conn = sqlite3.connect("database.db")
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS predictions
(credit REAL, age REAL, balance REAL, salary REAL, result TEXT)""")
conn.commit()
conn.close()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET","POST"])
def predict():

    result = None
    prob = None

    def safe(x):
        return float(x) if x and x.strip() != "" else 0

    if request.method == "POST":

        credit  = safe(request.form.get("credit"))
        gender  = safe(request.form.get("gender"))
        age     = safe(request.form.get("age"))
        tenure  = safe(request.form.get("tenure"))
        balance = safe(request.form.get("balance"))
        products = safe(request.form.get("products"))
        card    = safe(request.form.get("card"))
        active  = safe(request.form.get("active"))
        salary  = safe(request.form.get("salary"))
        germany = safe(request.form.get("germany"))
        spain   = safe(request.form.get("spain"))

        data = pd.DataFrame([[credit, gender, age, tenure, balance,
                              products, card, active, salary, germany, spain]],
                            columns=["CreditScore","Gender","Age","Tenure",
                                     "Balance","NumOfProducts","HasCrCard",
                                     "IsActiveMember","EstimatedSalary",
                                     "Geography_Germany","Geography_Spain"])

        p = model.predict_proba(data)[0][1]
        pred = model.predict(data)[0]

        prob = round(p*100,2)

        result = "Customer Likely To Leave" if pred==1 else "Customer Likely To Stay"

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO predictions VALUES (?,?,?,?,?)",
                  (credit, age, balance, salary, result))
        conn.commit()
        conn.close()

    return render_template("predict.html", result=result, prob=prob)


@app.route("/analytics")
def analytics():

    conn = sqlite3.connect("database.db")
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    leave = len(df[df["result"] == "Customer Likely To Leave"])
    stay  = len(df[df["result"] == "Customer Likely To Stay"])

    return render_template("analytics.html", leave=leave, stay=stay)


if __name__ == "__main__":
    app.run(debug=True)
