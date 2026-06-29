import pandas as pd
import numpy as np
import os
import argparse
import warnings
import multiprocessing as mp
import pickle
import os
import argparse
import warnings
import multiprocessing as mp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from regression_nonperiodic import (
    load_data, get_tickers_from_file, get_vectorized_metrics,
    STOX_FILE, START_CAPITAL, COMMISSION_RATE
)

warnings.filterwarnings('ignore')

# Global variables for multiprocessing workers to share memory efficiently (Copy-on-Write)
_global_raw_data = None
_global_features = ['slope', 'r2', 'score']
_global_hold_days = None

def init_worker(raw_data, hold_days):
    global _global_raw_data, _global_hold_days
    _global_raw_data = raw_data
    _global_hold_days = hold_days

def process_day(dt):
    """
    Worker function to process a single day.
    Strictly uses data up to 'dt' to avoid lookahead bias.
    """
    try:
        # 1. Isolate Data
        sliced_data = _global_raw_data.loc[:dt]
        if len(sliced_data) < 20: # Not enough data for metrics
            return dt, pd.DataFrame()
            
        # 2. Compute Metrics for the isolated data
        precalc = get_vectorized_metrics(sliced_data, lookback_days=20)
        
        # 3. Build Historical Matrix (Only trades where buy_dt + hold_days < dt)
        prices = precalc['prices']
        future_prices = prices.shift(-_global_hold_days)
        pl_pct_matrix = (future_prices - prices) / prices * 100
        
        slopes = precalc['slopes']
        r2 = precalc['r2']
        discounts = precalc['discounts']
        
        mask = (slopes > 0.0) & (r2 > 0.2) & (prices > 0)
        
        # Stack into Series
        valid_slopes = slopes[mask].stack()
        valid_r2 = r2[mask].stack()
        valid_score = discounts[mask].stack()
        valid_pl = pl_pct_matrix[mask].stack()
        
        df_hist = pd.DataFrame({
            'slope': valid_slopes,
            'r2': valid_r2,
            'score': valid_score,
            'pl_pct': valid_pl
        })
        
        df_hist.index.names = ['Date', 'Ticker']
        df_hist = df_hist.reset_index()
        
        # STRICT LOOKAHEAD PREVENTION: 
        # Only keep trades whose completion date is STRICTLY BEFORE dt
        # Because we already sliced `raw_data` up to `dt`, any trade that requires 
        # `future_prices` beyond `dt` will naturally be NaN in pl_pct.
        # dropna() correctly drops any 'future' trades that haven't closed yet.
        df_hist = df_hist.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df_hist) < 50:
            return dt, pd.DataFrame() # Not enough history to train KNN
            
        # 4. Train KNN
        scaler = StandardScaler()
        X_hist_scaled = scaler.fit_transform(df_hist[_global_features])
        
        K = min(50, len(df_hist))
        knn = NearestNeighbors(n_neighbors=K, algorithm='auto')
        knn.fit(X_hist_scaled)
        
        # 5. Predict candidates for today (dt)
        # Today is the last row in our sliced metrics
        idx = -1 
        today_slopes = slopes.iloc[idx]
        today_r2 = r2.iloc[idx]
        today_discounts = discounts.iloc[idx]
        today_prices = prices.iloc[idx]
        
        today_df = pd.DataFrame({
            'slope': today_slopes,
            'r2': today_r2,
            'score': today_discounts,
            'price': today_prices
        })
        
        today_df = today_df.replace([np.inf, -np.inf], np.nan).dropna()
        today_df = today_df[(today_df['slope'] > 0) & (today_df['r2'] > 0.1) & (today_df['price'] > 0)]
        
        if today_df.empty:
            return dt, pd.DataFrame()
            
        X_today_scaled = scaler.transform(today_df[_global_features])
        distances, indices = knn.kneighbors(X_today_scaled)
        
        expected_pl = []
        win_rates = []
        
        for j in range(len(today_df)):
            neighbor_indices = indices[j]
            neighbor_trades = df_hist.iloc[neighbor_indices]
            avg_pl = neighbor_trades['pl_pct'].mean()
            win_rate = (neighbor_trades['pl_pct'] > 0).mean() * 100
            expected_pl.append(avg_pl)
            win_rates.append(win_rate)
            
        today_df['exp_pl'] = expected_pl
        today_df['win_rate'] = win_rates
        today_df['confidence_score'] = (today_df['win_rate'] / 100) * today_df['exp_pl']
        
        return dt, today_df

    except Exception as e:
        print(f"\n[Worker Hata] {dt.date()}: {str(e)}")
        # Return empty DF on failure to avoid crashing the pool
        return dt, pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Portfolio Backtest using Sliding Window ML/KNN Predictions (NO OVERFITTING)")
    parser.add_argument('-n', '--num-stocks', type=int, default=1, help="Maksimum tutulacak hisse sayısı")
    parser.add_argument('-s', '--stop-loss', type=float, default=0.01, help="Zarar kesme (Trailing Stop) yüzdesi")
    parser.add_argument('-d', '--hold-days', type=int, default=10, help="Maksimum elde tutma gün sayısı")
    parser.add_argument('--months', type=int, default=6, help="Backtest süresi (Ay)")
    parser.add_argument('--cores', type=int, default=0, help="İşlemci çekirdeği sayısı (0 = tümü)")
    
    args = parser.parse_args()
    
    N_STOCKS = args.num_stocks
    STOP_LOSS = args.stop_loss
    MAX_HOLD_DAYS = pd.Timedelta(days=args.hold_days)
    MONTHS = args.months
    
    print(f"--- 🤖 SLIDING WINDOW PORTFOLIO BACKTEST BAŞLIYOR ---")
    print(f"Parametreler: N_STOCKS={N_STOCKS}, STOP_LOSS={STOP_LOSS*100}%, HOLD_DAYS={args.hold_days} gün, SÜRE={MONTHS} Ay")
    
    # 1. Market Verilerini Yükle
    print("\nBorsa verileri yükleniyor...")
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        return
        
    all_data = load_data(tickers)
    
    if isinstance(all_data.columns, pd.MultiIndex):
        try:
            raw_data = all_data['Close'].dropna(axis=1, how='all')
        except KeyError:
            raw_data = all_data.xs('Close', axis=1, level=0).dropna(axis=1, how='all')
    else:
        raw_data = all_data

    if not isinstance(raw_data.index, pd.DatetimeIndex):
        raw_data.index = pd.to_datetime(raw_data.index)
        all_data.index = pd.to_datetime(all_data.index)

    # 2. Tarihleri Belirle
    sim_start_date = (datetime.now() - timedelta(days=30 * MONTHS)).replace(hour=0, minute=0, second=0, microsecond=0)
    trading_days = raw_data.loc[sim_start_date:].index
    
    if len(trading_days) == 0:
        print("Geçerli işlem günü bulunamadı!")
        return

    # 3. PARALEL İŞLEME (ML Tahminleri)
    print(f"\n🧠 Adım 1: {len(trading_days)} işlem günü için bağımsız (look-ahead izoleli) öğrenme matrisleri oluşturuluyor...")
    # Güvenli performans sınırı (İşlemcinin aşırı yorulmasını önlemek için %60 civarı kullanım)
    safe_cores = max(1, int(mp.cpu_count() * 0.6))
    num_cores = args.cores if args.cores > 0 else safe_cores
    print(f"Paralel işleme {num_cores} çekirdek ile başlatıldı. Lütfen bekleyin, bu işlem bilgisayarınızı zorlayabilir...")
    
    daily_predictions = {}
    
    with mp.Pool(processes=num_cores, initializer=init_worker, initargs=(all_data, args.hold_days)) as pool:
        # Use imap to show progress bar
        results = list(tqdm(pool.imap(process_day, trading_days), total=len(trading_days), desc="ML Processing"))
        
    for dt, df in results:
        daily_predictions[dt] = df

    # Optimizasyon (Pitch) için matrisi kaydet
    dump_file = "sliding_window_predictions.pkl"
    with open(dump_file, "wb") as f:
        pickle.dump({
            "hold_days": args.hold_days,
            "predictions": daily_predictions
        }, f)
    print(f"✅ ML Matrisi ({dump_file}) başarıyla kaydedildi. Optimizasyon scripti bu dosyayı kullanacak.")

    # 4. SİMÜLASYON
    print("\n📈 Adım 2: Sıralı Portföy Simülasyonu başlatılıyor...")
    current_cash = START_CAPITAL
    active_portfolio = [] 
    trade_history = []
    deviations = []
    daily_vals = pd.Series(index=trading_days, dtype=float)

    for dt in tqdm(trading_days, desc="Daily Simulation"):
        # A. MEVCUT PORTFÖYÜ KONTROL ET (SATIŞLAR)
        for item in active_portfolio[:]:
            current_price = raw_data.at[dt, item['t']]
            if pd.isna(current_price) or current_price <= 0:
                continue
                
            should_sell = False
            reason = ""
            
            time_held = dt - item['buy_dt']
            if time_held >= MAX_HOLD_DAYS:
                should_sell = True
                reason = f"SÜRE DOLDU ({args.hold_days}G)"
            
            if not should_sell:
                if current_price > item['max_p']:
                    item['max_p'] = current_price
                    
                if current_price <= item['max_p'] * (1 - STOP_LOSS):
                    should_sell = True
                    reason = "TRAILING STOP"
            
            if should_sell:
                revenue = item['l'] * current_price * (1 - COMMISSION_RATE)
                current_cash += revenue
                pl_pct = (current_price / item['b'] - 1) * 100
                deviation = pl_pct - item.get('exp_pl', 0)
                deviations.append(deviation)
                
                trade_history.append([
                    dt.strftime('%Y-%m-%d'),
                    item['t'].replace('.IS', ''),
                    item['l'],
                    f"{current_price:.2f}",
                    reason,
                    f"{current_cash:,.2f}",
                    f"P/L: %{pl_pct:.2f} | Dev: %{deviation:.2f} | Peak: {item['max_p']:.2f}"
                ])
                
                active_portfolio.remove(item)
            else:
                item['days_held'] += 1

        # B. YENİ ALIMLAR (Eğer boş yer varsa)
        empty_slots = N_STOCKS - len(active_portfolio)
        if empty_slots > 0 and current_cash > 0:
            today_df = daily_predictions.get(dt, pd.DataFrame())
            
            if not today_df.empty:
                # Portföydeki hisseleri hariç tut
                active_tickers = [item['t'] for item in active_portfolio]
                today_df = today_df.drop(index=active_tickers, errors='ignore')
                
                best_candidates = today_df[(today_df['exp_pl'] > 0) & (today_df['win_rate'] >= 50)]
                best_candidates = best_candidates.sort_values(by='confidence_score', ascending=False)
                
                top_picks = best_candidates.head(empty_slots)
                
                if not top_picks.empty:
                    cash_per_slot = current_cash / empty_slots
                    for ticker, row in top_picks.iterrows():
                        buy_price = row['price']
                        lots = int(cash_per_slot / buy_price)
                        
                        if lots > 0:
                            cost = lots * buy_price
                            total_cost = cost * (1 + COMMISSION_RATE)
                            if total_cost > current_cash:
                                lots -= 1
                                cost = lots * buy_price
                                total_cost = cost * (1 + COMMISSION_RATE)
                                
                            if lots > 0:
                                current_cash -= total_cost
                                active_portfolio.append({
                                    't': ticker,
                                    'l': lots,
                                    'b': buy_price,
                                    'max_p': buy_price,
                                    'buy_dt': dt,
                                    'days_held': 0,
                                    'exp_pl': row['exp_pl']
                                })
                                
                                trade_history.append([
                                    dt.strftime('%Y-%m-%d'),
                                    ticker.replace('.IS', ''),
                                    lots,
                                    f"{buy_price:.2f}",
                                    "ALIS",
                                    f"{current_cash:,.2f}",
                                    f"Win: %{row['win_rate']:.1f} | ExpPL: %{row['exp_pl']:.2f}"
                                ])

        # C. GÜNLÜK DEĞERLEME
        port_val = current_cash
        for item in active_portfolio:
            cp = raw_data.at[dt, item['t']]
            if pd.isna(cp) or cp <= 0:
                cp = item['b']
            port_val += item['l'] * cp
            
        daily_vals[dt] = port_val

    # 5. SONUÇ RAPORU
    final_balance = daily_vals.iloc[-1]
    roi = ((final_balance - START_CAPITAL) / START_CAPITAL) * 100
    
    print(f"\n🎯 Sonuç: {START_CAPITAL:,.0f} TL -> {final_balance:,.2f} TL")
    print(f"Toplam Getiri: %{roi:.2f}")
    
    if deviations:
        print(f"Ortalama Sapma (Deviation): %{np.mean(deviations):.2f}")
        print(f"Minimum Sapma (Deviation): %{np.min(deviations):.2f}")
        print(f"Maksimum Sapma (Deviation): %{np.max(deviations):.2f}")

    # 6. EXCEL ÇIKTISI
    columns = ["Tarih", "Hisse", "Lot", "Fiyat", "İşlem", "Nakit", "Bilgi"]
    df_history = pd.DataFrame(trade_history, columns=columns)
    excel_file = "portfolio_backtest_fixed_results.xlsx"
    
    try:
        import openpyxl
        from openpyxl.styles import PatternFill
        
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Backtest Sonuçları"
        ws.append(columns)
        
        green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
        
        for index, row in df_history.iterrows():
            ws.append(row.tolist())
            current_row = ws[index + 2]
            
            is_buy = row['İşlem'] == 'ALIS'
            info = str(row['Bilgi'])
            action = str(row['İşlem'])
            
            fill = None
            if is_buy:
                fill = white_fill
            elif 'P/L: %-' in info:
                 fill = red_fill
            elif 'P/L: %' in info:
                 fill = green_fill
            elif 'SÜRE DOLDU' in action:
                 fill = green_fill
                 
            if fill:
                for cell in current_row:
                    cell.fill = fill
                    
        wb.save(excel_file)
        print(f"\nİşlem geçmişi '{excel_file}' dosyasına renkli olarak kaydedildi.")
    except ImportError:
        df_history.to_csv(excel_file.replace('.xlsx', '.csv'), index=False, sep=';', encoding='utf-8-sig')
        print(f"\nİşlem geçmişi '{excel_file.replace('.xlsx', '.csv')}' dosyasına kaydedildi.")

    # 7. GÖRSELLEŞTİRME
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(daily_vals.index, daily_vals.values, color='cyan', lw=2, label="ML Portfolio Strategy (Fixed/Sliding Window)")
    plt.axhline(y=START_CAPITAL, color='white', ls='--', alpha=0.3)
    plt.title(f"Sliding Window Backtest | Maks Hisse: {N_STOCKS} | Stop Loss: {STOP_LOSS*100}%", fontsize=14)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f} TL'))
    plt.legend()
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    
    chart_file = "portfolio_backtest_fixed_chart.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Portföy grafiği '{chart_file}' dosyasına kaydedildi.")

if __name__ == '__main__':
    # freeze_support required for Windows multiprocessing
    mp.freeze_support()
    main()
