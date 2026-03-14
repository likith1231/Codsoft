from flask import Flask, render_template, request
import pickle
import sqlite3
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# DB create
conn = sqlite3.connect("spam.db")
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS messages
(id INTEGER PRIMARY KEY AUTOINCREMENT,
text TEXT,
result TEXT,
confidence REAL)""")
conn.commit()
conn.close()


@app.route("/", methods=["GET","POST"])
def home():

    result = None
    confidence = None
    message = ""

    if request.method == "POST":

        message = request.form["message"]

        vec = vectorizer.transform([message])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)

        confidence = round(np.max(prob)*100,2)

        if pred == 1:
            result = "Spam Message"
        else:
            result = "Legitimate Message"

        conn = sqlite3.connect("spam.db")
        c = conn.cursor()
        c.execute("INSERT INTO messages(text,result,confidence) VALUES(?,?,?)",
                  (message,result,confidence))
        conn.commit()
        conn.close()

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           message=message)


@app.route("/history")
def history():

    conn = sqlite3.connect("spam.db")
    c = conn.cursor()
    c.execute("SELECT * FROM messages ORDER BY id DESC")
    data = c.fetchall()
    conn.close()

    return render_template("history.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)
