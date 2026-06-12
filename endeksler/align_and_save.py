import pandas as pd
import json
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_file = os.path.join(script_dir, "kap_endeks_referans.json")
    excel_file = os.path.join(script_dir, "Endeksler.xlsx")
    output_file = os.path.join(script_dir, "endeks_hisseleri.json")
    
    if not os.path.exists(ref_file):
        print(f"Hata: {ref_file} bulunamadı!")
        return
        
    with open(ref_file, "r", encoding="utf-8") as f:
        referans_liste = json.load(f)
        
    # Eşleşmeyi kolaylaştırmak için anahtarları büyük harfe çevirelim
    isimden_tickera = {item["Endeks Adı"].strip().upper(): item["Ticker"].strip() for item in referans_liste}
    
    if not os.path.exists(excel_file):
        print(f"Hata: {excel_file} bulunamadı!")
        return
        
    print(f"{excel_file} okunuyor ve dönüştürülüyor...")
    # Dosyayı başlıksız okuyoruz
    df = pd.read_excel(excel_file, header=None)
    
    endeks_hisseleri = {}
    eslesemeyenler = set()
    
    current_index_ticker = None
    
    # Her satırı dön
    for _, row in df.iterrows():
        # Sütunları temizleyelim
        col0 = str(row[0]).strip() if pd.notna(row[0]) else ""
        col1 = str(row[1]).strip() if pd.notna(row[1]) else ""
        
        # Boş satırları veya dosya başlığı olan 'Endeksler Listesi', 'Sıra' vb. atlayalım
        if not col0 and not col1:
            continue
        if col0.lower() in ["endeksler listesi", "sıra"]:
            continue
            
        # 1. DURUM: Kod sütunu boşsa, bu satır bir Endeks Başlığıdır (Örn: 'BIST 100')
        if col0 and not col1:
            current_index_name = col0.upper()
            
            # Referans tablosundan kısa Ticker kodunu (XU100) bulalım
            index_ticker = isimden_tickera.get(current_index_name)
            
            # Tam eşleşme yoksa substring ile bulmaya çalış
            if not index_ticker:
                for ref_isim, ref_ticker in isimden_tickera.items():
                    if ref_isim in current_index_name or current_index_name in ref_isim:
                        index_ticker = ref_ticker
                        break
                        
            if not index_ticker:
                eslesemeyenler.add(col0)
                
            # Bulunduysa Ticker'ı kullan, bulunamadıysa orijinal ismini kullan
            current_index_ticker = index_ticker if index_ticker else col0
            
            # Sözlükte bu endeks için bir liste başlatalım
            if current_index_ticker not in endeks_hisseleri:
                endeks_hisseleri[current_index_ticker] = []
                
        # 2. DURUM: Kod sütunu doluysa, bu bir Hissedir (Örn: 'AGHOL', 'AKBNK')
        elif col1 and col1.lower() not in ["kod", "ticker", "nan"]:
            if current_index_ticker:
                if col1 not in endeks_hisseleri[current_index_ticker]:
                    endeks_hisseleri[current_index_ticker].append(col1)

    # 3. JSON olarak kaydet
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(endeks_hisseleri, f, ensure_ascii=False, indent=4)
        
    print("\nİşlem Tamamlandı!")
    print(f"Toplam {len(endeks_hisseleri)} farklı endeks bulundu ve hisseleriyle eşleştirildi.")
    
    if eslesemeyenler:
        print("\nDikkat: Aşağıdaki endeks isimleri referans tablosunda eşleşmediği için orijinal adıyla kaydedildi:")
        for e in eslesemeyenler:
            print(" -", e)
            
    print(f"\nSonuçlar başarıyla '{output_file}' dosyasına kaydedildi!")

if __name__ == "__main__":
    main()
