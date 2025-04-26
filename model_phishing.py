import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# تحميل البيانات
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

X = df['URL']
y = df['label']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تهيئة الvectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5, stop_words='english', max_features=50)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# تدريب النموذج
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# اختبار النموذج
accuracy = model.score(X_test_tfidf, y_test)
print(f"دقة النموذج: {accuracy * 100:.2f}%")

# حفظ النموذج
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# حفظ الvectorizer
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)
