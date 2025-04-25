from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# تحميل البيانات وتدريب النموذج
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")  # بدون مسار مباشر
df = df.drop(columns=[
    'FILENAME', 'Domain', 'Title', 'LineOfCode', 'LargestLineLength', 'HasTitle', 'HasFavicon',
    'Robots', 'IsResponsive', 'NoOfPopup', 'NoOfiFrame', 'HasExternalFormSubmit',
    'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField',
    'Bank', 'Pay', 'Crypto'
])
X = df['URL']
y = df['label']

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5,
                             stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_tfidf, y)

# إعداد Flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url = request.form["url"]
        url_tfidf = vectorizer.transform([url])
        prediction = model.predict(url_tfidf)
        result = "Phishing" if prediction[0] == 1 else "Safe"
    return render_template("index.html", result=result)

# تشغيل التطبيق على Render
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

