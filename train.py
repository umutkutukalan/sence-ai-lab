import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Veriyi okuyalım
df = pd.read_csv('data_v1.csv')

# Vektörleştirici (Kelimeleri sayısal hale getirelim)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Modeli eğitelim
model = MultinomialNB()
model.fit(X, y)

# Manuel olarak test edelim
test_text = ["Yapay zekada ilk adımlarımı duygusal anlamlarla atmaya başladığım için heyecanlıyım."]
text_vector = vectorizer.transform(test_text)
result = model.predict(text_vector)

print(f"Test cümlesi: '{test_text[0]}'")
print(f"Tahmin edilen duygu: {result[0]}")