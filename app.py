from flask import Flask, render_template, request, redirect, session
import pickle
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)
app.secret_key = "simple-secret"

# Load ML model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# LOGIN PAGE
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["name"] = request.form["name"]
        session["email"] = request.form["email"]
        session["logged_in"] = True
        return redirect("/detect")

    return render_template("login.html")

# NEWS DETECTION PAGE
@app.route("/detect", methods=["GET", "POST"])
def detect():
    if not session.get("logged_in"):
        return redirect("/")

    result = ""
    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)
        result = "REAL NEWS" if pred[0] == 1 else "FAKE NEWS"

    return render_template("detect.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
