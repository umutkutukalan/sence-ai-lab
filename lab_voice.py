import pandas as pd
import speech_recognition as sr
from fuzzywuzzy import fuzz
import os

def listen_and_process():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # Ortam sesini 1 saniye boyunca dinleyelim ve kalibre edelim
        r.adjust_for_ambient_noise(source, duration=1)
        print("Dinliyorum... Şimdi söyleyin.")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
        
    try:
        # Google Speech Recognition API'sini kullanarak sesi metne dönüştürelim
        text = r.recognize_google(audio, language='en-US')
        return text.lower()
    except sr.UnknownValueError:
        return ["Anlaşılamayan ses. Lütfen tekrar deneyin."]
    except sr.RequestError:
        return ["API hizmetine erişilemiyor. Lütfen internet bağlantınızı kontrol edin."]
    
def run_lab():
    csv_path = "data/vocabs.csv"
    if not os.path.exists(csv_path):
        print(f"Veri dosyası bulunamadı: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print("--- SENSE-AI: TELAFFUZ LAB AKTIF ---")
    
    while True:
        # Rastgele bir kelime seçelim
        row = df.sample(n=1).iloc[0]
        target = str(row['word']).strip().lower()
        mean = row['translation']
        
        print(f"\nTarget Word: {target.upper()}")    
        print(f"Meaning: {mean}")
        print("Konuşmak için [Enter] tuşuna basın...")
        
        user_input = listen_and_process()
        print(f"Sizin söylediğiniz: {user_input}")
        
        # Benzerlik oranını hesaplayalım: Levenshtein tekniği ile
        score = fuzz.ratio(target, user_input)
        
        if score >= 90:
            print(f"🎉 Harika! (Skor: {score}%)")
        elif score >= 70:
            print(f"👍 İyi! (Skor: {score}%) - Tekrar Dene")
        else:
            print(f"👎 Daha iyi olabilir. (Skor: {score}%) - Tekrar Dene")
        
        if input("Tekrar denemek ister misiniz? (E/H): ").strip().lower() != 'e':
            print("Labdan çıkılıyor. Görüşmek üzere!")
            break

if __name__ == "__main__":
    run_lab()
        
        
    