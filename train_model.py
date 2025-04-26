# نستورد المكتبات
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# نقرأ ملف البيانات
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")  # اسم ملفك

# حذف الأعمدة النصية اللي تسبب مشاكل
for col in df.columns:
    if df[col].dtype == 'object':
        if col != 'label':  # نخلي عمود النتيجة فقط
            df = df.drop(col, axis=1)

# تحويل العمود 'label' إلى أرقام
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# نجهز البيانات
X = df.drop("label", axis=1)
y = df["label"]

# نقسم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ننشئ النموذج
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# نحفظ النموذج داخل ملف
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎉 تم حفظ النموذج بنجاح بدون مشاكل نصية!")
