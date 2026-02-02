import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timedelta
from scipy.stats import linregress
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# ==========================================================
# AYARLAR
# ==========================================================
STOX_FILE = 'top_endeks_hisseleri.txt'
DATA_CACHE_FILE = 'bist_data_cache.pkl'
START_CAPITAL = 100000
COMMISSION_RATE = 0.005
REBALANCE_FREQ = '7D'  # 7 Günlük Periyot
LOOKBACK_DAYS = 20  # Regresyon için geriye dönük gün sayısı
MIN_R_SQUARED = 0.50 # Regresyon uyum kalitesi (0-1 arası)
MIN_SLOPE = 0.0200  # Günlük asgari büyüme hızı
STOP_LOSS_RATE = 0.02 # %10 Stop Loss
MAX_ATR_PERCENT = 0.1 # Yüzdesel oynaklık limiti (ATR/Fiyat)
SLOPE_STOP_FACTOR = 0.0 # Günlük Asgari Getiri Oranı (Örn: 0.005 = Günlük %0.5 artış beklentisi)


def get_tickers_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} dosyası bulunamadı!")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [t.strip().upper() + ".IS" for t in f.read().splitlines() if t.strip()]

def load_data(tickers, force_refresh=False):
    sim_start = (datetime.now() - timedelta(days=365)).replace(day=1)
    download_start = (sim_start - timedelta(days=400)).strftime('%Y-%m-%d')
    
    cache_exists = os.path.exists(DATA_CACHE_FILE)
    if cache_exists and not force_refresh:
        # Cache dosyasının son değiştirilme tarihini kontrol et
        mtime = datetime.fromtimestamp(os.path.getmtime(DATA_CACHE_FILE))
        # Eğer bugün güncellendiyse cache'den yükle
        if mtime.date() == datetime.now().date():
            data = pd.read_pickle(DATA_CACHE_FILE)
            if hasattr(data, 'columns') and 'Volume' in data.columns:
                print("Güncel veriler cache'den yüklendi.")
                return data
    
    print(f"Yeni veriler indiriliyor (yfinance)... {'(Force Refresh)' if force_refresh else ''}")
    data = yf.download(tickers, start=download_start, auto_adjust=True)
    
    if not data.empty:
        # Hatalı/Eksik hisseleri tespit et
        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Close']
        else:
            closes = data
            
        # Tüm sütunları NaN olan veya hiç inmeyen hisseleri bul
        downloaded_tickers = closes.columns.tolist()
        failed_tickers = [t for t in tickers if t not in downloaded_tickers or closes[t].isna().all()]
        
        if failed_tickers:
            print(f"\n{len(failed_tickers)} hisse indirilemedi, tek tek deneniyor...")
            for t in tqdm(failed_tickers, desc="Eksik veriler tamamlanıyor"):
                try:
                    # Tekli indirme (threads=False daha stabil)
                    t_data = yf.download(t, start=download_start, auto_adjust=True, progress=False, threads=False)
                    if not t_data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            # MultiIndex yapısına uygun şekilde ekle
                            for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
                                if col in t_data.columns:
                                    data.loc[:, (col, t)] = t_data[col]
                        else:
                            data[t] = t_data['Close']
                except Exception as e:
                    # print(f"{t} hata: {e}")
                    pass
        
        # Sonuçları kaydet
        data.sort_index(axis=1, inplace=True)
        data.to_pickle(DATA_CACHE_FILE)
    
    return data

def get_clean_data(data):
    """
    Verideki terminal NaN satırlarını temizler.
    Eğer son satırda çok fazla NaN varsa, o satırı siler.
    """
    if data is None or data.empty:
        return data
    
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
        
    # Son satırdaki NaN oranını kontrol et
    while not closes.empty:
        last_row_nans = closes.iloc[-1].isna().sum()
        total_cols = len(closes.columns)
        
        # Eğer %50'den fazlası NaN ise bu satırı geçersiz say (örün market henüz açılmamış veya veri eksik)
        if last_row_nans > total_cols * 0.5:
            data = data.iloc[:-1]
            if isinstance(data.columns, pd.MultiIndex):
                closes = data['Close']
            else:
                closes = data
        else:
            break
            
    return data

def get_vectorized_metrics(all_data, lookback_days):
    """
    Tüm hisseler için regresyon ve ATR metriklerini vektörize olarak hesaplar.
    O(n) karmaşıklığında rolling window kullanarak performansı artırır.
    """
    if isinstance(all_data.columns, pd.MultiIndex):
        closes = all_data['Close'].dropna(axis=1, how='all')
        highs = all_data['High']
        lows = all_data['Low']
        volumes = all_data['Volume']
    else:
        closes = all_data
        highs = all_data.get('High', closes)
        lows = all_data.get('Low', closes)
        volumes = all_data.get('Volume', pd.DataFrame(0, index=closes.index, columns=closes.columns))

    n = lookback_days
    log_closes = np.log(closes)
    
    # X değerleri (0,1,2...,n-1)
    x = np.arange(n)
    sum_x = x.sum()
    sum_x2 = (x**2).sum()
    stdev_x = x.std()
    
    # Rolling Sums
    sum_y = log_closes.rolling(window=n).sum()
    sum_xy = log_closes.rolling(window=n).apply(lambda y: (x * y).sum(), raw=True)
    
    # 1. Slope (Eğim) Hesaplama
    # slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    slopes = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    
    # 2. Intercept (Kesişim) Hesaplama
    # intercept = (sum_y - slope * sum_x) / n
    intercepts = (sum_y - slopes * sum_x) / n
    
    # 3. R-Squared (Uyum Kalitesi) Hesaplama
    # Korelasyon katsayısı r = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_x2 - sum_x^2) * (n*sum_y2 - sum_y^2))
    sum_y2 = (log_closes**2).rolling(window=n).sum()
    numerator = (n * sum_xy - sum_x * sum_y)
    denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    r_squareds = (numerator / denominator)**2
    
    # 4. ATR Oranı (%) Hesaplama
    pc = closes.shift(1)
    tr = np.maximum(
        highs - lows,
        np.maximum((highs - pc).abs(), (lows - pc).abs())
    )
    
    atr = tr.rolling(14).mean()
    atr_pcts = atr / closes
    
    # 5. İskonto (%) Hesaplama
    # expected_log_price = intercept + slope * (n - 1)
    expected_log_prices = intercepts + slopes * (n - 1)
    expected_prices = np.exp(expected_log_prices)
    discounts = (expected_prices - closes) / expected_prices
    
    vol_avgs = volumes.rolling(21).mean()
    
    # Forward fill (ffill) ekleyerek terminal NaN'ları (örneğin seans başı/sonu veri eksikliği) tolere ediyoruz.
    # Bu, Tab 3'teki .dropna() yöntemine benzer şekilde her hissenin en son valid verisini kullanmasını sağlar.
    limit = 5
    return {
        'slopes': slopes.ffill(limit=limit),
        'intercepts': intercepts.ffill(limit=limit),
        'r2': r_squareds.ffill(limit=limit),
        'atr_pct': atr_pcts.ffill(limit=limit),
        'discounts': discounts.ffill(limit=limit),
        'volumes': volumes.ffill(limit=limit),
        'vol_avgs': vol_avgs.ffill(limit=limit),
        'prices': closes.ffill(limit=limit)
    }

def find_best_candidate(target_date, all_data, lookback_days=LOOKBACK_DAYS, max_atr_percent=MAX_ATR_PERCENT, 
                        min_slope=MIN_SLOPE, min_r2=MIN_R_SQUARED, precalc=None):
    # Eğer precalc varsa, vektörize veriden direkt çek
    if precalc is not None:
        try:
            # En yakın tarihi bul (Borsa tatil olabilir)
            idx = precalc['prices'].index.get_indexer([target_date], method='pad')[0]
            if idx < 0: return []
            
            dt = precalc['prices'].index[idx]
            
            # Tüm hisseler için maske oluştur
            mask = (precalc['slopes'].iloc[idx] > min_slope) & \
                   (precalc['r2'].iloc[idx] > min_r2) & \
                   (precalc['atr_pct'].iloc[idx] <= max_atr_percent)
            
            valid_tickers = mask[mask].index.tolist()
            candidates = []
            
            for ticker in valid_tickers:
                candidates.append({
                    't': ticker,
                    'slope': precalc['slopes'].at[dt, ticker],
                    'r2': precalc['r2'].at[dt, ticker],
                    'score': precalc['discounts'].at[dt, ticker],
                    'price': precalc['prices'].at[dt, ticker],
                    'vol_avg': precalc['vol_avgs'].at[dt, ticker],
                    'vol_curr': precalc['volumes'].at[dt, ticker]
                })
            
            # İskonto (skore) göre sırala
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates
        except Exception as e:
            # print(f"Vectorized search error: {e}")
            return []

    # Eğer precalc None ise veya fonksiyon buraya düşerse boş liste dön
    return []

if __name__ == "__main__":
    # 1. VERİ YÜKLEME
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        exit()
    
    all_data = load_data(tickers)
    
    # Veri Ayrıştırma (Simülasyon İçin Loop Dışında Lazım)
def run_simulation(all_data, lookback_days=LOOKBACK_DAYS, min_slope=MIN_SLOPE, min_r2=MIN_R_SQUARED, 
                   stop_loss_rate=STOP_LOSS_RATE, slope_stop_factor=SLOPE_STOP_FACTOR,
                   rebalance_freq=REBALANCE_FREQ, start_capital=START_CAPITAL, commission_rate=COMMISSION_RATE,
                   max_atr_percent=MAX_ATR_PERCENT, progress_callback=None, silent=False):
                   
    # Veri Ayrıştırma
    if isinstance(all_data.columns, pd.MultiIndex):
        try:
            raw_data = all_data['Close'].dropna(axis=1, how='all')
            raw_volume = all_data['Volume']
        except KeyError:
            raw_data = all_data.xs('Close', axis=1, level=0).dropna(axis=1, how='all')
            raw_volume = all_data.xs('Volume', axis=1, level=0)
    else:
        raw_data = all_data
        if 'Volume' in all_data.columns:
            raw_volume = all_data['Volume']
        else:
            raw_volume = pd.DataFrame(0, index=raw_data.index, columns=raw_data.columns)
            
    sim_start_date = (datetime.now() - timedelta(days=365)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    periods = pd.date_range(start=sim_start_date, end=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0), freq=rebalance_freq)

    daily_vals = pd.Series(index=raw_data.loc[periods[0]:].index, dtype=float)
    trade_history = []
    current_cash = start_capital
    active_portfolio = [] # [{'t': ticker, 'l': lots, 'b': buy_price}]

    # Vektörize Metrikleri Hesapla (Hız için)
    precalc = get_vectorized_metrics(all_data, lookback_days)
    
    # İlk değerleri doldur
    for d in daily_vals.index:
        daily_vals[d] = start_capital

    total_periods = len(periods) - 1

    for i in tqdm(range(total_periods), desc="Regression Backtest", disable=silent):
        if progress_callback:
            progress_callback((i + 1) / total_periods)
        # En yakın geçerli işlem gününü bul
        start_idx = raw_data.index.get_indexer([periods[i]], method='nearest')[0]
        end_idx = raw_data.index.get_indexer([periods[i+1]], method='nearest')[0]
        
        start_date = raw_data.index[start_idx]
        end_date = raw_data.index[end_idx]
        
        # --- PORTFÖY SATIŞ (Dönem Başı) ---
        stock_revenue = 0
        if active_portfolio:
            # Mevcut hisseleri sat
            for item in active_portfolio:
                current_price = raw_data.at[start_date, item['t']]
                if pd.isna(current_price) or current_price <= 0:
                    current_price = item['b']
                
                val = item['l'] * current_price * (1 - commission_rate)
                stock_revenue += val
                
                pl_pct = (current_price / item['b'] - 1) * 100
                trade_history.append([
                    start_date.strftime('%Y-%m-%d'), item['t'].replace('.IS',''), item['l'],
                    f"{current_price:.2f}", "SATIS", f"{(current_cash + stock_revenue):,.2f}", 
                    f"P/L: %{pl_pct:.2f} (REBALANCE)"
                ])
            
            current_cash += stock_revenue
            active_portfolio = []

        # --- REGRESYON ANALİZİ & SEÇİM ---
        # Vektörize aday seçimi
        candidates = find_best_candidate(start_date, all_data, lookback_days, max_atr_percent, min_slope, min_r2, precalc=precalc)
        # Filtreleme burada tekrar yapılabilir mi? Hayır, zaten filtered geliyor.
        
        top_pick = candidates[0] if candidates else None
        
        # --- ALIM YAP ---
        if top_pick:
            buy_price = top_pick['price']
            if buy_price > 0:
                lots = int(current_cash / buy_price)
                if lots > 0:
                    cost = lots * buy_price
                    current_cash -= cost * (1 + commission_rate)
                    active_portfolio.append({
                        't': top_pick['t'], 
                        'l': lots, 
                        'b': buy_price, 
                        'max_p': buy_price, 
                        'slope': top_pick['slope'], 
                        'buy_dt': start_date,
                        'days_held': 0
                    })
                    
                    trade_history.append([
                        start_date.strftime('%Y-%m-%d'), top_pick['t'].replace('.IS',''), lots,
                        f"{buy_price:.2f}", "ALIS", f"{current_cash:,.2f}", 
                        f"(Eğim: {top_pick['slope']:.4f}, R2: {top_pick['r2']:.2f})"
                    ])
        
        # --- GÜNLÜK DEĞERLEME ---
        period_prices = raw_data.loc[start_date:end_date]
        
        for dt in period_prices.index:
            # 1. Stop Loss Kontrolü
            for item in active_portfolio[:]:
                curr_p = period_prices.at[dt, item['t']]
                if pd.isna(curr_p) or curr_p <= 0:
                    continue
                
                # Max Fiyat Güncelleme (Trailing Stop İçin)
                if curr_p > item['max_p']:
                    item['max_p'] = curr_p

                # Fiyat Stop (Trailing Stop: Max fiyattan %X düşerse sat)
                if curr_p <= item['max_p'] * (1 - stop_loss_rate):
                    sell_val = item['l'] * curr_p * (1 - commission_rate)
                    current_cash += sell_val
                    active_portfolio.remove(item)
                    pl_pct = (curr_p / item['b'] - 1) * 100
                    trade_history.append([
                        dt.strftime('%Y-%m-%d'), item['t'].replace('.IS',''), item['l'],
                        f"{curr_p:.2f}", "STOP LOSS", f"{current_cash:,.2f}", 
                        f"P/L: %{pl_pct:.2f} | Peak: {item['max_p']:.2f}"
                    ])
                    continue
                
                # 2. Günlük Minimum Getiri Stopu (Fixed Daily Return Stop)
                if slope_stop_factor != 0:
                    # slope_stop_price = buy_price * (1 + daily_return_target)^days_held
                    slope_stop_p = item['b'] * ((1 + slope_stop_factor) ** item['days_held'])
                    
                    if curr_p < slope_stop_p:
                        sell_val = item['l'] * curr_p * (1 - commission_rate)
                        current_cash += sell_val
                        active_portfolio.remove(item)
                        pl_pct = (curr_p / item['b'] - 1) * 100
                        trade_history.append([
                            dt.strftime('%Y-%m-%d'), item['t'].replace('.IS',''), item['l'],
                            f"{curr_p:.2f}", "SLOPE STOP", f"{current_cash:,.2f}", 
                            f"P/L: %{pl_pct:.2f} | Floor: {slope_stop_p:.2f}"
                        ])
                        continue

                # Gün sayısını artır
                item['days_held'] += 1

            # 3. Portföy Değeri
            port_value = current_cash
            for item in active_portfolio:
                curr_p = period_prices.at[dt, item['t']]
                if pd.isna(curr_p) or curr_p <= 0:
                    curr_p = item['b']
                port_value += item['l'] * curr_p
            
            daily_vals[dt] = port_value

    # Sonucu Döndür
    daily_vals = daily_vals.loc[:end_date]
    return daily_vals, trade_history, daily_vals.iloc[-1]

if __name__ == "__main__":
    # 1. VERİ YÜKLEME
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        exit()
    
    all_data = load_data(tickers)
    
    # 2. SİMÜLASYON BAŞLAT
    print(f"--- Simülasyon Başlıyor (Regresyon: {LOOKBACK_DAYS} Gün) ---")
    daily_vals, trade_history, final_balance = run_simulation(all_data)


    # SON DURUM RAPORU
    # Simülasyonun bittiği yere kadar kes (Son dönem sonrası veriler 19000 kalmış olabilir)
    # daily_vals = daily_vals.loc[:end_date]  <-- REMOVED: end_date is undefined here and daily_vals is already sliced inside run_simulation
    
    final_balance = daily_vals.iloc[-1]
    roi = ((final_balance - START_CAPITAL) / START_CAPITAL) * 100
    
    print(f"\n🎯 Sonuç: {START_CAPITAL:,.0f} TL -> {final_balance:,.2f} TL")
    print(f"Toplam Getiri: %{roi:.2f}")
    
    # --- GÖRSELLEŞTİRME ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    
    # Ana Grafik
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(daily_vals.index, daily_vals.values, color='cyan', lw=2, label="Exponential Regression Model")
    ax1.axhline(y=START_CAPITAL, color='white', ls='--', alpha=0.3)
    ax1.set_title(f"Üslü Regresyon Stratejisi | Lookback: {LOOKBACK_DAYS} Gün | R2 >= {MIN_R_SQUARED}", fontsize=14)
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f} TL'))
    ax1.legend()
    ax1.grid(True, alpha=0.15)
    
    # İşlem Tablosu
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    columns = ["Tarih", "Hisse", "Lot", "Fiyat", "İşlem", "Nakit", "Bilgi"]
    table_data = trade_history[-15:] if trade_history else [["-"] * 7]
    the_table = ax2.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.5)
    
    for (row, col), cell in the_table.get_celld().items():
        cell.set_text_props(color='black', fontweight='bold')
        if row == 0:
            cell.set_facecolor('#3498db')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#ecf0f1')
    
    plt.tight_layout()
    plt.show()
