import joblib
from flask import Flask, render_template, request
import re
from urllib.parse import urlparse

app = Flask(__name__)

# تحميل النموذج المدرب
model = joblib.load("phishing_model.pkl")
print("✅ تم تحميل النموذج")

# دالة استخراج الخصائص من الرابط
def extract_features(url):
    features = []
    features.append(len(url))  # طول الرابط
    features.append(url.count('@'))  # عدد @
    features.append(url.count('.'))  # عدد .
    features.append(url.count('-'))  # عدد -
    features.append(url.count('='))  # عدد =
    features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)  # يستخدم IP أو لا
    parsed = urlparse(url)
    features.append(len(parsed.netloc))  # طول الدومين
    return features

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        user_input = request.form.get('url_input')
        if not user_input:
            result = "الرجاء إدخال رابط."
        else:
            try:
                # استخراج الخصائص من الرابط
                features = extract_features(user_input)
                prediction = model.predict([features])
                result = "Phishing" if prediction[0] == 1 else "Not Phishing"
            except Exception as e:
                print(f"❌ خطأ أثناء المعالجة: {e}")
                result = "حدث خطأ أثناء معالجة الرابط."
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
