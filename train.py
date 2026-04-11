import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Veriyi okuyalım
df = pd.read_csv('data_v1.csv')

# Etiketlerdeki boşlukları temizleyelim ve hepsini küçük harfe çevirelim
df['label'] = df['label'].str.strip().str.lower()
df['text'] = df['text'].str.lower()

# Vektörleştirici (Kelimeleri sayısal hale getirelim)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Veriyi eğitim ve test olarak bölelim (örneğin %80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitelim
model = MultinomialNB()
model.fit(X_train, y_train)

# Ölçüm yapalım
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"--- Model Performansı ---")
print(f"Başarı Oranı: %{accuracy * 100:.2f}")
print(f"Rapor:\n{classification_report(y_test, predictions)}")

# Canlı test yapalım
sample = ["Bu akşamki konser harikaydı, çok eğlendim!"]
sample_vectorized = vectorizer.transform(sample)
print(f"\nÖrnek Test: '{sample[0]}' -> Tahmin: {model.predict(sample_vectorized)[0]}")