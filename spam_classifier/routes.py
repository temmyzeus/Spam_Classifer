from spam_classifier import app
from flask import render_template, request
import pickle
from spam_classifier import config

data_transformer = pickle.load(
    open(config.data_transform_name, "rb"))
model = pickle.load(open(config.model_name, "rb"))

@ app.route("/")
@ app.route("/index")
@ app.route("/home")
def home():
    return render_template("index.html", title="Spam & Ham Classifier")

@ app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        text = [str(request.form.get("predict_text", None))]
        text_tfidf=data_transformer.transform(text)
        outcome = model.predict(text_tfidf)[0]
        if outcome == 1:
            outcome = "Spam"
        elif outcome == 0:
            outcome = "Ham"
        return render_template("result.html", outcome=outcome)