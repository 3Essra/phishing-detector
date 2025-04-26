import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import re
from urllib.parse import urlparse
import joblib

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

# قراءة الداتا
df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')

# تجهيز البيانات
X = df['URL'].apply(extract_features).tolist()
y = df['Label'].apply(lambda x: 1 if x == 'Phishing' else 0)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestClassifier()
model.fit(X_train, y_train)

# حفظ النموذج
joblib.dump(model, 'phishing_model.pkl')

# تقييم الدقة
accuracy = model.score(X_test, y_test)
print(f"دقة النموذج: {accuracy * 100:.2f}%")
