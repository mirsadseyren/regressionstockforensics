#https://www.kap.org.tr/tr/endeksler

import undetected_chromedriver as uc
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Tarayıcı başlatılıyor...")
    driver = uc.Chrome(headless=False, use_subprocess=True)
    driver.set_page_load_timeout(30)

    url = "https://www.kap.org.tr/tr/Endeksler"
    driver.get(url)

    try:
        print("Sayfanın yüklenmesi bekleniyor...")
        # Elementlerin yüklenmesi için bekle
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//span[@class='font-semibold']"))
        )
        time.sleep(3) # Ekstra dinamik içerik yüklenmesi için kısa bir bekleme

        # Kullanıcının belirttiği elemente tıklayarak listeyi aç (Genişlet)
        try:
            dropdown = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "(//div[contains(@class,'cursor-pointer flex')])[1]"))
            )
            dropdown.click()
            print("Belirtilen sekmeye tıklandı ve liste genişletildi.")
            time.sleep(2) # Animasyonun tamamlanması için bekle
        except Exception as e:
            print(f"Sekmeye tıklanırken hata oluştu: {e}")
        
        # Kullanıcının belirttiği XPath'leri kullanarak tüm elementleri bul
        names_elements = driver.find_elements(By.XPATH, "//span[@class='font-semibold']")
        tickers_elements = driver.find_elements(By.XPATH, "//span[@class='font-medium']")
        
        print(f"Toplam bulunan başlık sayısı: {len(names_elements)}")
        print(f"Toplam bulunan ticker sayısı: {len(tickers_elements)}")
        
        reference_table = []
        
        # Notlarda belirtilen başlangıç indexleri:
        # (//span[@class='font-semibold'])[2] -> Python listesinde index 1
        # (//span[@class='font-medium'])[3]   -> Python listesinde index 2
        name_idx = 1
        ticker_idx = 2
        
        while name_idx < len(names_elements) and ticker_idx < len(tickers_elements):
            # Element görünür olmasa bile metnini çekmek için textContent kullanıyoruz
            name = names_elements[name_idx].get_attribute("textContent").strip()
            ticker = tickers_elements[ticker_idx].get_attribute("textContent").strip()
            
            # Eğer ikisi de doluysa listeye ekle
            if name and ticker:
                reference_table.append({
                    "Endeks Adı": name,
                    "Ticker": ticker
                })
            
            name_idx += 1
            ticker_idx += 1

        # Sonuçları ekrana aynı satırda sırayla yazdır
        print("\n" + "="*80)
        print(f"{'ENDEKS ADI':<60} | {'TICKER'}")
        print("="*80)
        for row in reference_table:
            print(f"{row['Endeks Adı']:<60} | {row['Ticker']}")
        print("="*80)
            
        # Veriyi JSON dosyasına kaydet (scriptin kendi dizinine)
        output_path = os.path.join(script_dir, "kap_endeks_referans.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(reference_table, f, ensure_ascii=False, indent=4)
            
        print(f"\nToplam {len(reference_table)} adet eşleşme bulundu.")
        print("Veriler başarıyla 'kap_endeks_referans.json' dosyasına kaydedildi.")

    except Exception as e:
        print(f"Hata oluştu: {e}")

    finally:
        driver.quit()
        print("Tarayıcı kapatıldı.")

if __name__ == "__main__":
    main()
