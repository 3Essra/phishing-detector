import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# تحميل النموذج والمدرب و vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
    print("✅ تم تحميل النموذج")

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)
    print("✅ تم تحميل الvectorizer")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        user_input = request.form.get('url_input')
        if not user_input:
            result = "الرجاء إدخال رابط."
        else:
            try:
                # استخدام نفس الvectorizer اللي تدرب عليه النموذج
                input_vector = vectorizer.transform([user_input])
                prediction = model.predict(input_vector)
                result = "Phishing" if prediction[0] == 1 else "Not Phishing"
            except Exception as e:
                print(f"❌ خطأ أثناء المعالجة: {e}")
                result = "حدث خطأ أثناء معالجة الرابط."
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
