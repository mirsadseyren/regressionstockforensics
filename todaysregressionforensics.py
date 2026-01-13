from regression import load_data, get_tickers_from_file, find_best_candidate, STOX_FILE
from datetime import datetime, timedelta
import pandas as pd
import warnings
import json

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print("--- GÜNLÜK REGRESYON TARAMASI ---")
    
    dategap = int(input("Tarih Farkı (gün): "))
    
    # 1. Hisseleri Yükle
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        print("Hisse listesi bulunamadı!")
        exit()
        
    # 2. Verileri Yükle (Cache'den veya taze indir)
    # Cache dosyasının güncelliğini kontrol edelim, eski ise yeniden indirsin
    # Şimdilik load_data cache kullanıyor, eğer güncel veri yoksa cache silinmeli.
    # Kullanıcı "bugünün" verisini istediği için cache'i silsen iyi olur ama yavaşlatır.
    # Pratik çözüm: load_data cache kullanır, kullanıcı taze veri isterse cache dosyasını siler.
    print("Veriler yükleniyor...")
    all_data = load_data(tickers)
    
    # 3. Son İşlem Gününü Bul
    if isinstance(all_data.columns, pd.MultiIndex):
        try:
            last_date = all_data['Close'].index[-1] - timedelta(days=dategap)
        except KeyError:
            last_date = all_data.xs('Close', axis=1, level=0).index[-1]
    else:
        last_date = all_data.index[-1]
        
    print(f"Analiz Tarihi: {datetime.now().date() - timedelta(days=dategap)}")
    
    # 4. Adayları Bul
    candidates = find_best_candidate(last_date, all_data)
    
    print(f"\nBulunan Aday Sayısı: {len(candidates)}")
    print("-" * 90)
    print(f"{'TICKER':<15} {'FİYAT':<10} {'EĞİM':<10} {'R2':<8} {'SKOR':<8} {'CURR_VOL':<15} {'AVG_VOL':<15}")
    print("-" * 90)
    
    for c in candidates[:10]: # İlk 10 tanesini göster
        ticker = c['t'].replace('.IS', '')
        price = c['price']
        slope = c['slope']
        r2 = c['r2']
        score = c['score']
        vol_curr = c['vol_curr']
        vol_avg = c['vol_avg']
        
        print(f"{ticker:<15} {price:<10.2f} {slope:<10.4f} {r2:<8.2f} {score:<8.4f} {vol_curr:<15,.0f} {vol_avg:<15,.0f}")
        
    print("-" * 90)
    if candidates:
        print(f"🎯 EN İYİ SEÇİM: {candidates[0]['t'].replace('.IS', '')}")
    else:
        print("Uygun aday bulunamadı.")
    json.dump(candidates, open('candidates.json', 'w'), indent=4)
