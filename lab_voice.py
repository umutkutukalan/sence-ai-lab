import pandas as pd
import speech_recognition as sr
from fuzzywuzzy import fuzz
import os
import webrtcvad
import numpy as np

def is_speech(audio_data, sample_rate=16000, aggressiveness=1):
    """
    Bu fonksiyon, verilen ses verisinin konuşma içerip içermediğini belirlemek için VAD (Voice Activity Detection) tekniğini kullanır.
    
    Parametreler:
    - audio_data: Ses verisi (bytes formatında)
    - sample_rate: Ses örnekleme hızı (varsayılan: 16000 Hz)
    - aggressiveness: VAD'nin ne kadar agresif olduğunu belirler (0-3 arası, 3 en agresif)
    
    Dönüş Değeri:
    - True: Ses verisi konuşma içeriyor
    - False: Ses verisi konuşma içermiyor
    """
    
    vad = webrtcvad.Vad(aggressiveness)
    # Ses verisini 30 ms'lik parçalar halinde bölüyoruz, insan konuşmasımı diye kontrol etmek için
    frame_duration = 30  # ms
    # Burada / yerine // kullanarak tam sayı bölmesi yapıyoruz, böylece tam byte sayısı elde ederiz
    samples_per_frame = (sample_rate * frame_duration // 1000)
    bytes_per_frame = samples_per_frame * 2  # 16-bit audio için 2 byte per sample
    
    
    # Sadece ilk parçaya değil, ilk 10 parçadan herhangi birinde ses var mı diye bak
    for i in range(0, min(len(audio_data), bytes_per_frame * 10), bytes_per_frame):
        frame = audio_data[i:i + bytes_per_frame]
        if len(frame) == bytes_per_frame:
            if vad.is_speech(frame, sample_rate):
                return True
    return False


def listen_and_process():
    r = sr.Recognizer()
    r.energy_threshold = 300
    
    with sr.Microphone(sample_rate=16000) as source:
        print("\n>>> Lab Kalibrasyonu yapılıyor...")
        r.adjust_for_ambient_noise(source, duration=0.8)
        print(">>> Dinliyorum... (Kelimeyi söyleyene kadar pusudayım)")

        while True:
            try:
                # timeout=None: Sen konuşana kadar bekler
                audio = r.listen(source, timeout=None, phrase_time_limit=4)
                raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                
                # Eğer VAD hata verirse veya ses gelmezse döngü kırılmasın diye try-except içine aldık
                try:
                    if is_speech(raw_data, sample_rate=16000, aggressiveness=1):
                        text = r.recognize_google(audio, language="en-US")
                        if text:
                            return text.lower()
                except Exception:
                    continue # VAD hatası olursa sessizce dinlemeye devam et
                    
            except sr.UnknownValueError:
                continue
            except Exception as e:
                # Cihaz bağlantısı gibi kritik hataları bildir
                return f"[⚠️ Kritik Hata: {str(e)}]"
    
    
    
def run_lab():
    csv_path = "data/vocabs.csv"
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı!")
        return

    df = pd.read_csv(csv_path)
    print("--- SENSE-AI: TELAFFUZ LAB AKTIF ---")

    while True:
        # Rastgele bir kelime seç
        row = df.sample(n=1).iloc[0]
        target = str(row['word']).strip().lower()
        mean = row['translation']

        # Bu iç döngü, kullanıcı kelimeyi doğru dürüst söyleyene kadar aynı kelimede tutar
        while True:
            print(f"\nTarget Word: {target.upper()}")
            print(f"Meaning: {mean}")
            input("Konuşmak için [ENTER] tuşuna basın...")

            user_input = listen_and_process()

            # Eğer None döndüyse veya liste/hata formatındaysa güvenliğe al
            if not user_input or not isinstance(user_input, str):
                user_input = "[Sistem hatası: Geçersiz veri]"

            # --- KORUYUCU KATMAN ---
            # Eğer konuşma algılanmadıysa veya gürültü elendiyse
            if user_input.startswith("["):
                print(f"⚠️  {user_input}")
                print(">>> Değerlendirme yapılmadı. Lütfen tekrar deneyin.")
                continue # AYNI kelime için tekrar input bekler

            # Sadece geçerli bir input geldiyse puanlamaya geçer
            print(f"Sizin söylediğiniz: {user_input}")
            score = fuzz.ratio(target, user_input)

            if score >= 90:
                print(f"✅ MÜKEMMEL! (Skor: {score})")
                break # Bu kelime bitti, ana döngüye dönüp yeni kelime seçer
            elif score >= 70:
                print(f"🟡 YAKIN (Skor: {score}) - Bir kez daha deneyebilirsin.")
                if input("Tekrar denemek ister misin? (e/h): ").lower() == 'h':
                    break
            else:
                print(f"❌ HATALI (Skor: {score})")
                if input("Tekrar denemek ister misin? (e/h): ").lower() == 'h':
                    break

        if input("\nSonraki kelimeye geçmek için [ENTER], çıkmak için [q]: ").lower() == 'q':
            break

if __name__ == "__main__":
    run_lab()
        
        
    