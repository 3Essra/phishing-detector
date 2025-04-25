import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# تحميل البيانات من ملف CSV
df = pd.read_csv(r"C:\PhiUSIIL_Phishing_URL_Dataset.csv")  # استبدلي 'path_to_file' بمسار الملف الفعلي

# عرض الأعمدة المتاحة في البيانات
print(df.columns)

# إزالة الأعمدة غير الضرورية
df = df.drop(columns=[
    'FILENAME', 'Domain', 'Title', 'LineOfCode', 'LargestLineLength', 
    'HasTitle', 'HasFavicon', 'Robots', 'IsResponsive', 'NoOfPopup', 
    'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton', 
    'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto'
])

# إعداد X (النصوص) و y (التصنيفات)
X = df['URL']  # النصوص من عمود URL
y = df['label']  # التصنيفات من عمود label

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحسين TfidfVectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # الكلمات المفردة والثنائية
    max_df=0.9,  # تجاهل الكلمات التي تظهر في أكثر من 90% من الوثائق
    min_df=5,  # تجاهل الكلمات التي تظهر في أقل من 5 وثائق
    stop_words='english',  # تجاهل الكلمات الشائعة في الإنجليزية
    max_features=5000  # الاحتفاظ فقط بـ 5000 كلمة الأكثر أهمية
)

# تحويل النصوص إلى تمثيلات عددية باستخدام TfidfVectorizer
X_train_tfidf = vectorizer.fit_transform(X_train)  # تدريب المحول على بيانات التدريب
X_test_tfidf = vectorizer.transform(X_test)  # تحويل بيانات الاختبار باستخدام المحول المدرب

# بناء نموذج Naive Bayes
model = MultinomialNB()

# تدريب النموذج باستخدام البيانات المحولة
model.fit(X_train_tfidf, y_train)

# اختبار النموذج
accuracy = model.score(X_test_tfidf, y_test)
print(f"دقة النموذج: {accuracy * 100:.2f}%")
