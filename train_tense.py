import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/tense.csv", encoding='latin-1')

df.columns = df.columns.str.strip() # Sütun isimlerindeki boşlukları temizleyelim

print(f"--Veri Dağılımı (Sınıf Bazlı)")
print(df['tense'].value_counts())

df.dropna(subset=['sentence'], inplace=True)
df['sentence'] = df['sentence'].astype(str).str.lower()

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3), 
    max_features=5000,
    analyzer='word' # Burayı 'char_wb' yaparak ekleri daha iyi yakalamasını sağlayabiliriz ama şimdilik 'word' kalsın.
)
X = vectorizer.fit_transform(df['sentence'])
y = df['tense']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, C=10)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"\nGenel Başarı Oranı: %{accuracy_score(y_test, predictions) * 100:.2f}")
print("\n--- Detaylı Sınıflandırma Raporu ---")
print(classification_report(y_test, predictions))

stres_testi = [
    "The findings of the research will have been analyzed by the time the conference starts.", # Future Perfect
    "Scientists are currently investigating the long-term effects of climate change.", # Present Continuous
    "The Roman Empire expanded significantly during the reign of Augustus.", # Past Simple
    "Researchers have recently discovered a new species in the Amazon rainforest." # Present Perfect
]

for cumle in stres_testi:
    vec = vectorizer.transform([cumle.lower()])
    tahmin = model.predict(vec)[0]
    print(f"Cümle: {cumle} -> Tahmin: {tahmin}")