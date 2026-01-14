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
STOX_FILE = 'stox.txt'
DATA_CACHE_FILE = 'bist_data_cache.pkl'
START_CAPITAL = 19000
COMMISSION_RATE = 0.002
REBALANCE_FREQ = '5D'  # 5 Günlük Periyot
LOOKBACK_DAYS = 30  # Regresyon için geriye dönük gün sayısı
MIN_R_SQUARED = 0.50 # Regresyon uyum kalitesi (0-1 arası) - Düşürüldü
MIN_SLOPE = 0.02   # Günlük asgari büyüme hızı
STOP_LOSS_RATE = 0.10 # %15 Stop Loss
MAX_ATR_PERCENT = 0.08 # Yüzdesel oynaklık limiti (ATR/Fiyat)
VOLUME_STOP_RATIO = 5.0 # Hacim ortalamasının kaç katına çıkarsa satılsın

def get_tickers_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} dosyası bulunamadı!")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [t.strip().upper() + ".IS" for t in f.read().splitlines() if t.strip()]

def load_data(tickers):
    sim_start = (datetime.now() - timedelta(days=365)).replace(day=1)
    download_start = (sim_start - timedelta(days=400)).strftime('%Y-%m-%d')
    
    if os.path.exists(DATA_CACHE_FILE):
        data = pd.read_pickle(DATA_CACHE_FILE)
        if hasattr(data, 'columns') and 'Volume' in data.columns:
            return data
    
    print("Veriler indiriliyor (Close + Volume)...")
    data = yf.download(tickers, start=download_start, auto_adjust=True)
    data.to_pickle(DATA_CACHE_FILE)
    return data

def find_best_candidate(target_date, all_data, lookback_days=LOOKBACK_DAYS, max_atr_percent=MAX_ATR_PERCENT, 
                        min_slope=MIN_SLOPE, min_r2=MIN_R_SQUARED):
    # Veri Hazırlığı
    if isinstance(all_data.columns, pd.MultiIndex):
        try:
            raw_data = all_data['Close'].dropna(axis=1, how='all')
            raw_high = all_data['High']
            raw_low = all_data['Low']
            raw_volume = all_data['Volume']
        except KeyError:
            raw_data = all_data.xs('Close', axis=1, level=0).dropna(axis=1, how='all')
            raw_high = all_data.xs('High', axis=1, level=0)
            raw_low = all_data.xs('Low', axis=1, level=0)
            raw_volume = all_data.xs('Volume', axis=1, level=0)
    else:
        raw_data = all_data
        if 'High' in all_data.columns:
            raw_high = all_data['High']
            raw_low = all_data['Low']
            raw_volume = all_data['Volume']
        else:
            raw_high = raw_data
            raw_low = raw_data
            raw_volume = pd.DataFrame(0, index=raw_data.index, columns=raw_data.columns)
            
    available_tickers = raw_data.columns.tolist()
    candidates = []
    
    # O tarihte işlem gören ve yeterli geçmişi olan hisseleri bul
    analysis_start = target_date - timedelta(days=lookback_days * 1.5)
    window_data = raw_data.loc[analysis_start:target_date]
    
    for ticker in available_tickers:
        # Son N günlük veri (NaN temizlenmiş)
        series = window_data[ticker].dropna().tail(lookback_days)
        
        if len(series) < lookback_days * 0.8: # En az %80 veri olsun
            continue
        
        # --- ATR HESAPLAMA (Volatilite Filtresi) ---
        try:
            # İlgili dönemin High/Low verilerini al
            s_high = raw_high[ticker].loc[series.index]
            s_low = raw_low[ticker].loc[series.index]
            s_close = series
            
            # TR (True Range) hesapla
            ph = s_high
            pl = s_low
            pc = s_close.shift(1)
            
            tr = pd.concat([
                ph - pl,
                (ph - pc).abs(),
                (pl - pc).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(14).mean().iloc[-1]
            last_price = s_close.iloc[-1]
            
            atr_pct = atr / last_price
            
            # Aşırı oynaksa ele (Kara Liste)
            if atr_pct > max_atr_percent:
                continue
                
        except:
            # Veri eksikse (High/Low yoksa) filtreyi pas geç veya ele
            continue
            
        # Exponential Regression: ln(y) = ln(a) + bx
        # y = Fiyatlar, x = 0, 1, 2...
        try:
            y = np.log(series.values)
            x = np.arange(len(y))
            
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # Slope = Günlük logaritmik getiri (~yüzdesel büyüme)
            # R_squared = r_value ** 2 (Uyum kalitesi)
            
            if slope > min_slope and (r_value ** 2) > min_r2:
                # Beklenen Fiyat (Exponential: y = exp(intercept + slope * x))
                expected_log_price = intercept + slope * (len(y) - 1)
                expected_price = np.exp(expected_log_price)
                current_price = series.iloc[-1]
                
                # İskonto Oranı: (Beklenen - Aktüel) / Beklenen
                discount = (expected_price - current_price) / expected_price
                
                # Score: Yeni kurala göre discount (negatif olabilir, ama biz pozitif discountları arıyoruz)
                score = discount
                
                # Volume Stats
                try:
                    # target_date dahil geriye dönük 21 gün
                    v_series = raw_volume[ticker].loc[:target_date].tail(21)
                    vol_avg = v_series.mean()
                    vol_curr = v_series.iloc[-1]
                except:
                    vol_avg = 0
                    vol_curr = 0
                
                candidates.append({
                    't': ticker,
                    'slope': slope,
                    'r2': r_value ** 2,
                    'score': score,
                    'price': series.iloc[-1],
                    'vol_avg': vol_avg,
                    'vol_curr': vol_curr
                })
        except Exception as e:
            continue
            
    # Sıralama (En yüksek skorlu 1 hisseyi alalım - Momentum stratejisi gibi konsantre)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates

if __name__ == "__main__":
    # 1. VERİ YÜKLEME
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        exit()
    
    all_data = load_data(tickers)
    
    # Veri Ayrıştırma (Simülasyon İçin Loop Dışında Lazım)
def run_simulation(all_data, lookback_days=LOOKBACK_DAYS, min_slope=MIN_SLOPE, min_r2=MIN_R_SQUARED, 
                   volume_stop_ratio=VOLUME_STOP_RATIO, stop_loss_rate=STOP_LOSS_RATE,
                   rebalance_freq=REBALANCE_FREQ, start_capital=START_CAPITAL, commission_rate=COMMISSION_RATE,
                   max_atr_percent=MAX_ATR_PERCENT):
                   
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
            
    # Volum Ortalaması (21 Gün)
    vol_avg = raw_volume.rolling(21).mean()

    sim_start_date = (datetime.now() - timedelta(days=365)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    periods = pd.date_range(start=sim_start_date, end=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0), freq=rebalance_freq)

    daily_vals = pd.Series(index=raw_data.loc[periods[0]:].index, dtype=float)
    trade_history = []
    current_cash = start_capital
    active_portfolio = [] # [{'t': ticker, 'l': lots, 'b': buy_price}]

    # İlk değerleri doldur
    for d in daily_vals.index:
        daily_vals[d] = start_capital

    total_periods = len(periods) - 1

    for i in tqdm(range(total_periods), desc="Regression Backtest"):
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
        # Burada simülasyon parametreleri (min_slope vs) find_best_candidate'e geçirilmiyor
        # ama find_best_candidate global sabitler kullanıyordu.
        # En temiz yol find_best_candidate'e parametre eklemek ama şimdilik global sabitleri güncelleyemeyiz.
        # Bu yüzden find_best_candidate'i de güncellememiz lazım ama şimdilik aynen çağırıyoruz.
        # NOT: find_best_candidate varsayılan LOOKBACK_DAYS'i argüman olarak alıyor, ancak slope/r2 threshold'ları hardcoded.
        # Simülasyon fonksiyonunda bunlara müdahale edememek sorun yaratabilir.
        # Şimdilik global değişkenleri atlayıp fonksiyonu çağırıyoruz, 
        # ancak find_best_candidate'i refactor etmemiz gerekecek.
        # Hızlı çözüm: find_best_candidate zaten refactor edildi, parametre alacak şekilde güncelleyelim?
        # Zaten lookback_days alıyor. min_slope ve min_r2'yi de ekleyelim.
        candidates = find_best_candidate(start_date, all_data, lookback_days, max_atr_percent, min_slope, min_r2)
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
                    active_portfolio.append({'t': top_pick['t'], 'l': lots, 'b': buy_price, 'max_p': buy_price})
                    
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
                
                # Hacim Stop
                try:
                    curr_vol = raw_volume.at[dt, item['t']]
                    avg_vol = vol_avg.at[dt, item['t']]
                    if not pd.isna(curr_vol) and not pd.isna(avg_vol) and avg_vol > 0:
                        if curr_vol > avg_vol * volume_stop_ratio:
                            sell_val = item['l'] * curr_p * (1 - commission_rate)
                            current_cash += sell_val
                            active_portfolio.remove(item)
                            trade_history.append([
                                dt.strftime('%Y-%m-%d'), item['t'].replace('.IS',''), item['l'],
                                f"{curr_p:.2f}", "HACIM STOP", f"{current_cash:,.2f}", 
                                f"Vol: {curr_vol/avg_vol:.1f}x Avg"
                            ])
                except:
                    pass
            
            # 2. Portföy Değeri
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
