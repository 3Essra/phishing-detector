from flask import Flask, render_template, request
import pickle

# تحميل النموذج والمتجه
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url = request.form["url"]
        features = vectorizer.transform([url])
        prediction = model.predict(features)[0]
        result = "Phishing" if prediction == 1 else "Safe"
    return render_template("index.html", result=result)

# تشغيل التطبيق على Render
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
