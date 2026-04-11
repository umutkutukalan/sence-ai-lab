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

# Basit bir Türkçe stop word listesi (daha kapsamlı bir liste kullanılabilir)
turkce_stop_words = [
    've', 'bir', 'bu', 'da', 'de', 'için', 'ile', 'gibi', 'ama', 'çok',
    'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'ne', 'mi', 'mı',
    # Daha fazla stop word eklenebilir
]

# Vektörleştirici (Kelimeleri sayısal hale getirelim)
vectorizer = TfidfVectorizer(
    stop_words=turkce_stop_words,
    ngram_range=(1, 2) # Hem tek kelimeleri hem de ikili kelime gruplarını dikkate alalım
)
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

# Stres Testi
sample = ["Backend tarafındaki karmaşık bugları temizledikten sonra gelen o hafifleme hissi paha biçilemez."]
sample_vectorized = vectorizer.transform(sample)
print(f"\nÖrnek Test: '{sample[0]}' -> Tahmin: {model.predict(sample_vectorized)[0]}")
