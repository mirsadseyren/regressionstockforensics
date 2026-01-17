
import os
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

# --- AYARLAR ---
CACHE_FILE = 'bist_data_cache.pkl'
STOX_FILE = 'stox.txt'
BASE_URL = "https://www.oyakyatirim.com.tr/Equity/GetHistoricalEquityData?mode=2&code={}"

def load_tickers(file_path):
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} dosyası bulunamadı!")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Kodun sonundaki .IS'i kaldırıp sadece kök sembolü alacağız
        return [t.strip().upper().replace('.IS', '') for t in f.read().splitlines() if t.strip()]

def get_oyak_data(ticker):
    url = BASE_URL.format(ticker)
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Veri formatı: [[timestamp_ms, price], ...]
            # Timestamp'i datetime'a çevirelim
            df = pd.DataFrame(data, columns=['Date', 'Close'])
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            
            # Günlük verilere çevir (saat farkı varsa normalize et)
            df['Date'] = df['Date'].dt.normalize()
            df.set_index('Date', inplace=True)
            
            # OHLCV formatına tamamla
            df['Open'] = df['Close']
            df['High'] = df['Close']
            df['Low'] = df['Close']
            df['Volume'] = 0.0 # Hacim verisi yok
            
            # Sütun sırasını düzenle
            # Indirme verisiyle uyumlu olması için: Close, Open, High, Low, Volume
            return df[['Close', 'Open', 'High', 'Low', 'Volume']]
        else:
            print(f"HATA ({ticker}): HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"HATA ({ticker}): {e}")
        return None

def patch_cache():
    print("--- DATA YAMA ISLEMI BASLATILIYOR (V2 - 0/NaN Kontrolü) ---")
    
    # 1. Cache Yükle
    if not os.path.exists(CACHE_FILE):
        print("Cache dosyası bulunamadı, önce regression.py çalıştırıp verileri indiriniz.")
        return

    try:
        cache_data = pd.read_pickle(CACHE_FILE)
        print("Mevcut veri seti yüklendi.")
    except Exception as e:
        print(f"Cache yüklenemedi: {e}")
        return

    # 2. Hisseleri Yükle
    tickers = load_tickers(STOX_FILE)
    print(f"Toplam {len(tickers)} hisse kontrol edilecek.")
    
    patches_applied = 0
    overwrites_applied = 0
    
    # MultiIndex kontrolü
    is_multiindex = isinstance(cache_data.columns, pd.MultiIndex)
    
    today = pd.Timestamp.now().normalize()
    check_start_date = today - pd.Timedelta(days=15)
    
    for ticker in tqdm(tickers, desc="Kontrol Ediliyor"):
        yf_ticker = ticker + ".IS"
        data_needs_patch = False
        
        # Bu hissenin verisine ulaş
        try:
            if is_multiindex:
                if yf_ticker in cache_data.columns.levels[1]:
                    stock_closes = cache_data['Close'][yf_ticker]
                else:
                    continue
            else:
                if yf_ticker in cache_data.columns:
                    stock_closes = cache_data[yf_ticker]
                else:
                    continue
        except KeyError:
            continue
            
        # --- Kontrol 1: Veri çok mu eski? ---
        valid_dates = stock_closes.dropna().index
        if valid_dates.empty:
            last_date = pd.Timestamp("2020-01-01")
        else:
            last_date = valid_dates[-1]
            
        days_diff = (today - last_date).days
        if days_diff > 1:
            data_needs_patch = True
            
        # --- Kontrol 2: Son 15 günde 0 veya NaN var mı? ---
        if not data_needs_patch:
            # Son 15 güne ait kısmı al (bugüne kadar, cache'deki son tarihe kadar değil, takvim gününe göre)
            # Cache'de bu tarihler hiç yoksa (NaN ise) veya 0 ise
            
            # Öncelikle cache'de bu aralığa denk gelen satırlar var mı ona bakalım
            # Create range of dates for last 15 days excluding weekends if possible, but simpler to just check index intersection
            
            recent_slice = stock_closes.loc[stock_closes.index >= check_start_date]
            
            # 0 veya NaN kontrolü
            if recent_slice.empty:
                # Veri yoksa patch gerekir (zaten days_diff yakalar ama yine de)
                pass 
            else:
                # 0 veya NaN olan gün var mı?
                has_zeros = (recent_slice == 0).any()
                has_nans = recent_slice.isna().any()
                
                if has_zeros or has_nans:
                    data_needs_patch = True
                    # print(f" {ticker} bozuk veri tespit edildi (0/NaN).")

        if data_needs_patch:
            # Oyak'tan çek
            new_data = get_oyak_data(ticker)
            
            if new_data is not None and not new_data.empty:
                ticker_patched = False
                
                for dt, row in new_data.iterrows():
                    # Sadece ilgilendiğimiz tarih aralığına bak (eski veriyi bozmayalım, ama son 1 ay güvenli)
                    if dt < check_start_date:
                        continue
                        
                    price = row['Close']
                    if price <= 0: continue # Oyak'ta da 0 ise yapacak bir şey yok
                    
                    # Cache'deki durumu kontrol et
                    # 1. Tarih cache'de yok mu? (Append)
                    # 2. Tarih var ama değer 0 veya NaN mı? (Overwrite)
                    
                    should_write = False
                    
                    if dt not in cache_data.index:
                        should_write = True
                    else:
                        try:
                            if is_multiindex:
                                current_val = cache_data.loc[dt, ('Close', yf_ticker)]
                            else:
                                current_val = cache_data.loc[dt, yf_ticker]
                                
                            if pd.isna(current_val) or current_val == 0:
                                should_write = True
                        except:
                            should_write = True
                            
                    if should_write:
                        if is_multiindex:
                            # Satır yoksa eklemek zor olabilir, reindex gerekebilir ama
                            # .loc var olmayan index'e yazarsa yeni satır ekler (warning verebilir)
                            cache_data.loc[dt, ('Close', yf_ticker)] = row['Close']
                            cache_data.loc[dt, ('Open', yf_ticker)] = row['Open']
                            cache_data.loc[dt, ('High', yf_ticker)] = row['High']
                            cache_data.loc[dt, ('Low', yf_ticker)] = row['Low']
                            cache_data.loc[dt, ('Volume', yf_ticker)] = 0.0
                        else:
                            cache_data.loc[dt, yf_ticker] = row['Close']
                        
                        ticker_patched = True
                        overwrites_applied += 1
                
                if ticker_patched:
                    patches_applied += 1
                    
    print(f"\nToplam {patches_applied} hisse üzerinde işlem yapıldı.")
    print(f"Toplam {overwrites_applied} adet gün verisi düzeltildi/eklendi.")
    
    # 3. Kaydet
    if patches_applied > 0:
        cache_data.sort_index(inplace=True)
        cache_data.to_pickle(CACHE_FILE)
        print(f"Data kaydı güncellendi: {CACHE_FILE}")
    else:
        print("Güncellenecek eksik/hatalı veri bulunamadı.")

if __name__ == "__main__":
    patch_cache()
