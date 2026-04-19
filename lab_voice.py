import pandas as pd
import speech_recognition as sr
from fuzzywuzzy import fuzz
import os

def listen_and_process():
    r = sr.Recognizer()
    
    # --- Hassasiyetli bir şekilde ortam sesini kalibre edelim ---
    # Bu adım, ortam gürültüsünü tanımak ve filtrelemek için önemlidir. Böylece kullanıcı konuşurken daha net bir şekilde algılanır.
    r.energy_threshold = 600  # Bu değeri ortam gürültüsüne göre ayarlayabilirsiniz. Düşükse, daha sessiz sesleri algılar; yüksekse, sadece daha yüksek sesleri algılar.
    
    r.pause_threshold = 0.8  # Konuşma arasındaki duraklamaları tanımak için bu değeri ayarlayabilirsiniz. Daha düşükse, daha kısa duraklamaları tanır; daha yüksekse, daha uzun duraklamaları tanır.
    
    r.dynamic_energy_threshold = True  # Bu özellik, ortam gürültüsüne göre otomatik olarak enerji eşiğini ayarlamaya çalışır. Genellikle açık bırakmak iyi sonuç verir.
    
    with sr.Microphone() as source:
        print("\n>>> Lab Kalibrasyonu: Arka plan gürültüsü eleniyor...")
        # 1 saniye yerine 0.5 saniye hızlıca o anki tıklama/fan sesini örnekle
        r.adjust_for_ambient_noise(source, duration=0.5)
        
        print("Dinliyorum... Şimdi söyleyin.")
        
        # Kullanıcının konuşmasını dinleyelim (maksimum 10 saniye)
        # timeout: Kullanıcının konuşmaya başlaması için beklenen maksimum süre (saniye cinsinden)
        # phrase_time_limit: Kullanıcının konuşmasının maksimum süresi (saniye cinsinden)
        audio = r.listen(source, timeout=100, phrase_time_limit=5)
        
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
        
        
    