import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import os
import csv

df_main = pd.read_csv("data/tense.csv", encoding='latin-1')
df_main.columns = df_main.columns.str.strip() # Sütun isimlerindeki boşlukları temizleyelim

ozel_veri_yolu = "data/data_ozel_yokdil.csv"

if os.path.exists(ozel_veri_yolu):
    #Not: Kaydederken hangi encoding'i kullandıysan
    df_ozel = pd.read_csv(ozel_veri_yolu, encoding='utf-16')
    df_ozel.columns = df_ozel.columns.str.strip() # Sütun isimlerindeki boşlukları temizleyelim
    
    # İki veri setini birleştirelim
    df = pd.concat([df_main, df_ozel], ignore_index=True)
    print(f"🔄 Özel veri seti yüklendi! Toplam veri: {len(df)}")
else:
    df = df_main
    print(f"📊 Sadece ana veri seti ile başlanıyor. Toplam veri: {len(df)}")

print(f"--Veri Dağılımı (Sınıf Bazlı)")
print(df_main['tense'].value_counts())

df = df.dropna(subset=['sentence'])
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
    
    
def save_to_custom_dataset(sentence, tense):
    file_path = "data/data_ozel_yokdil.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-16') as f:
        write = csv.writer(f)
        if not file_exists:
            write.writerow(['sentence', 'tense'])
        write.writerow([sentence, tense])
    
print("\n--- Sense-AI Eğitim Modu Aktif ---")
print("Çıkmak için 'q' yazabilirsiniz.")

while True:
    user_input = input("Analiz edilecek cümleyi girin: ")
    if user_input.lower() == 'q':
        break
    
    vec = vectorizer.transform([user_input.lower()])
    prediction = model.predict(vec)[0]
    
    print(f"Tahmin Edilen Zaman: {prediction}")
    feedback = input("Bu tahmin doğru mu? (e/h): ")
    
    if feedback == 'y':
        final_tense = prediction
    else:
        final_tense = input("Doğru tense nedir? (örn: past, future perfect, present perfect): ").lower()
        
    save_to_custom_dataset(user_input, final_tense)
    print("Veri kaydedildi. Teşekkürler!\n")