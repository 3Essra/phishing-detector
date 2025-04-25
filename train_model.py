
import pandas as pd
# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# تحميل البيانات
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
df = df.drop(columns=[
    'FILENAME', 'Domain', 'Title', 'LineOfCode', 'LargestLineLength', 'HasTitle', 'HasFavicon',
    'Robots', 'IsResponsive', 'NoOfPopup', 'NoOfiFrame', 'HasExternalFormSubmit',
    'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField',
    'Bank', 'Pay', 'Crypto'
])
X = df['URL']
y = df['label']

# تحويل النصوص
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5,
                             stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# تدريب النموذج
model = MultinomialNB()
model.fit(X_tfidf, y)

# حفظ النموذج والمتجه
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ تم حفظ النموذج والمتجه.")
