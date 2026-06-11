import pandas as pd
import numpy as np
import os
import argparse
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import yfinance as yf
import time

from regression_nonperiodic import (
    load_data, get_tickers_from_file, get_vectorized_metrics,
    STOX_FILE, START_CAPITAL, COMMISSION_RATE
)

warnings.filterwarnings('ignore')

def fetch_and_calculate_financial_score(tickers):
    print("\nTüm hisseler için yfinance finansal verileri çekiliyor (Bu işlem 1-2 dakika sürebilir)...")
    financial_data = []
    
    # Sadece .IS ile biten geçerli tickerları filtrele
    valid_tickers = [t for t in tickers if isinstance(t, str) and t.endswith('.IS')]
    
    for ticker in tqdm(valid_tickers, desc="Finansal Veri"):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            data = {
                'Ticker': ticker,
                'PD/DD (Price to Book)': info.get('priceToBook', np.nan),
                'FD/FAVÖK (EV/EBITDA)': info.get('enterpriseToEbitda', np.nan),
                'F/K (Trailing PE)': info.get('trailingPE', np.nan),
                'FAVÖK Marjı (EBITDA Margin)': info.get('ebitdaMargins', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'ROA': info.get('returnOnAssets', np.nan),
                'Ortalama Hacim (Avg Volume)': info.get('averageVolume', np.nan)
            }
            financial_data.append(data)
            time.sleep(0.05)
        except Exception:
            pass
            
    df = pd.DataFrame(financial_data).set_index('Ticker')
    metrics = df.columns.tolist()
    
    # Z-Score Normalization
    for col in metrics:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val != 0 and not pd.isna(std_val):
            df[col] = (df[col] - mean_val) / std_val
            
    # Korelasyon Ağırlıkları (correlation_backtest.py sonuçlarından)
    weights = {
        'PD/DD (Price to Book)': 0.86,
        'FD/FAVÖK (EV/EBITDA)': -14.83,
        'F/K (Trailing PE)': 0.63,
        'FAVÖK Marjı (EBITDA Margin)': 1.50,
        'ROE': -3.85,
        'ROA': -2.26,
        'Ortalama Hacim (Avg Volume)': 18.93
    }
    
    df['Finansal Güven Skoru'] = 0.0
    for metric in metrics:
        df['Finansal Güven Skoru'] += df[metric].fillna(0) * weights.get(metric, 0.0)
        
    return df['Finansal Güven Skoru'].to_dict()

def main():
    parser = argparse.ArgumentParser(description="Portfolio Backtest v2 using Finansal x Küme Güveni")
    parser.add_argument('-n', '--num-stocks', type=int, default=1, help="Maksimum tutulacak hisse sayısı (default: 1)")
    parser.add_argument('-s', '--stop-loss', type=float, default=0.06, help="Zarar kesme (Trailing Stop) yüzdesi (default: 0.09 = %%9)")
    parser.add_argument('-d', '--hold-days', type=int, default=10, help="Maksimum elde tutma gün sayısı (default: 16)")
    parser.add_argument('--min-conf', type=float, default=4.1, help="Minimum ML Confidence Score")
    parser.add_argument('--max-conf', type=float, default=10.1, help="Maksimum ML Confidence Score")
    parser.add_argument('--min-fin', type=float, default=0.0, help="Minimum Finansal Güven Skoru (Percentile 0-100)")
    parser.add_argument('--max-fin', type=float, default=100.0, help="Maksimum Finansal Güven Skoru (Percentile 0-100)")
    
    args = parser.parse_args()
    
    N_STOCKS = args.num_stocks
    STOP_LOSS = args.stop_loss
    MAX_HOLD_DAYS = pd.Timedelta(days=args.hold_days)
    MIN_CONF = args.min_conf
    MAX_CONF = args.max_conf
    MIN_FIN = args.min_fin
    MAX_FIN = args.max_fin
    
    print(f"--- 🤖 PORTFOLIO ML BACKTEST V2 BAŞLIYOR (FİNANSAL x KÜME GÜVENİ) ---")
    print(f"Parametreler: N_STOCKS={N_STOCKS}, STOP_LOSS={STOP_LOSS*100}%, HOLD_DAYS={args.hold_days} gün")
    print(f"Filtreler: ML_Conf_Skor:[{MIN_CONF}, {MAX_CONF}], Finansal_Skor:[{MIN_FIN}, {MAX_FIN}]")

    # 1. Load Historical Matrix and train KNN
    matrix_file = 'historical_trade_metrics.csv'
    if not os.path.exists(matrix_file):
        print(f"Hata: {matrix_file} bulunamadı. Lütfen önce 'analyze_trades.py' çalıştırın.")
        return

    print("Historical matrix yükleniyor ve KNN modeli eğitiliyor...")
    hist_df = pd.read_csv(matrix_file)
    features = ['slope', 'r2', 'score']
    
    scaler = StandardScaler()
    X_hist_scaled = scaler.fit_transform(hist_df[features])
    
    K = 50
    knn = NearestNeighbors(n_neighbors=K, algorithm='auto')
    knn.fit(X_hist_scaled)

    # 2. Market Verilerini Yükle
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

    # 2.1 Yfinance Finansal Skoru Hesapla
    financial_scores_dict = fetch_and_calculate_financial_score(raw_data.columns)

    # 1 Yıllık simülasyon başlangıcı
    sim_start_date = (datetime.now() - timedelta(days=365)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if not isinstance(raw_data.index, pd.DatetimeIndex):
        raw_data.index = pd.to_datetime(raw_data.index)
        
    trading_days = raw_data.loc[sim_start_date:].index
    
    if len(trading_days) == 0:
        print("Geçerli işlem günü bulunamadı!")
        return

    # Vektörize metrikler
    print("\nTeknik metrikler vektörize olarak hesaplanıyor...")
    precalc = get_vectorized_metrics(all_data, lookback_days=20)

    # 3. Simülasyon
    current_cash = START_CAPITAL
    active_portfolio = [] 
    trade_history = []
    deviations = []
    daily_vals = pd.Series(index=trading_days, dtype=float)

    total_days = len(trading_days)
    for i, dt in tqdm(enumerate(trading_days), total=total_days, desc="Daily Backtest"):
        idx = precalc['prices'].index.get_indexer([dt], method='pad')[0]
        if idx < 0:
            continue
            
        # A. MEVCUT PORTFÖYÜ KONTROL ET (SATIŞLAR)
        for item in active_portfolio[:]:
            current_price = raw_data.at[dt, item['t']]
            if pd.isna(current_price) or current_price <= 0:
                continue
                
            should_sell = False
            reason = ""
            
            # Zaman Stopu
            time_held = dt - item['buy_dt']
            if time_held >= MAX_HOLD_DAYS:
                should_sell = True
                reason = f"SÜRE DOLDU ({args.hold_days}G)"
            
            # Trailing Stop Loss
            if not should_sell:
                if current_price > item['max_p']:
                    item['max_p'] = current_price
                    
                if current_price <= item['max_p'] * (1 - STOP_LOSS):
                    should_sell = True
                    reason = "TRAILING STOP"
            
            # Satış İşlemi
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
            slopes = precalc['slopes'].iloc[idx]
            r2 = precalc['r2'].iloc[idx]
            discounts = precalc['discounts'].iloc[idx]
            prices = precalc['prices'].iloc[idx]
            
            today_df = pd.DataFrame({
                'slope': slopes,
                'r2': r2,
                'score': discounts,
                'price': prices
            })
            
            today_df = today_df[(today_df['slope'] > 0) & (today_df['r2'] > 0.1) & (today_df['price'] > 0)]
            active_tickers = [item['t'] for item in active_portfolio]
            today_df = today_df.drop(index=active_tickers, errors='ignore')
            
            if not today_df.empty:
                X_today_scaled = scaler.transform(today_df[features])
                distances, indices = knn.kneighbors(X_today_scaled)
                
                expected_pl = []
                win_rates = []
                
                for j in range(len(today_df)):
                    neighbor_indices = indices[j]
                    neighbor_trades = hist_df.iloc[neighbor_indices]
                    avg_pl = neighbor_trades['pl_pct'].mean()
                    win_rate = (neighbor_trades['pl_pct'] > 0).mean() * 100
                    expected_pl.append(avg_pl)
                    win_rates.append(win_rate)
                    
                today_df['exp_pl'] = expected_pl
                today_df['win_rate'] = win_rates
                today_df['confidence_score'] = today_df['exp_pl']
                
                # === FİNANSAL GÜVEN SKORU ENTEGRASYONU (Percentile Normalization) ===
                fin_scores = [financial_scores_dict.get(t, 0.0) for t in today_df.index]
                today_df['fin_skor_raw'] = fin_scores
                # BIST içindeki göreceli konumunu 0-100 arasına (yüzdelik dilim) oturt
                today_df['finansal_skor'] = today_df['fin_skor_raw'].rank(pct=True) * 100
                
                best_candidates = today_df[
                    (today_df['exp_pl'] > 0) & 
                    (today_df['win_rate'] >= 50) &
                    (today_df['confidence_score'] >= MIN_CONF) &
                    (today_df['confidence_score'] <= MAX_CONF) &
                    (today_df['finansal_skor'] >= MIN_FIN) &
                    (today_df['finansal_skor'] <= MAX_FIN)
                ]
                
                # Sadece ML Güven Skoru (Expected P/L) ile azalan sıralama
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
                                    f"Finansal: {row['finansal_skor']:.0f}/100 | ExpPL: %{row['exp_pl']:.2f}"
                                ])

        # C. GÜNLÜK DEĞERLEME
        port_val = current_cash
        for item in active_portfolio:
            cp = raw_data.at[dt, item['t']]
            if pd.isna(cp) or cp <= 0:
                cp = item['b']
            port_val += item['l'] * cp
            
        daily_vals[dt] = port_val

    # 4. SONUÇ RAPORU
    final_balance = daily_vals.iloc[-1]
    roi = ((final_balance - START_CAPITAL) / START_CAPITAL) * 100
    
    print(f"\n🎯 Sonuç: {START_CAPITAL:,.0f} TL -> {final_balance:,.2f} TL")
    print(f"Toplam Getiri: %{roi:.2f}")
    
    if deviations:
        print(f"Ortalama Sapma (Deviation): %{np.mean(deviations):.2f}")
        print(f"Minimum Sapma (Deviation): %{np.min(deviations):.2f}")
        print(f"Maksimum Sapma (Deviation): %{np.max(deviations):.2f}")

    # 5. EXCEL ÇIKTISI
    columns = ["Tarih", "Hisse", "Lot", "Fiyat", "İşlem", "Nakit", "Bilgi"]
    df_history = pd.DataFrame(trade_history, columns=columns)
    excel_file = "portfolio_backtest_v2_results.xlsx"
    
    try:
        import openpyxl
        from openpyxl.styles import PatternFill
        
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Backtest V2 Sonuçları"
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

    # 6. GÖRSELLEŞTİRME
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(daily_vals.index, daily_vals.values, color='magenta', lw=2, label="ML + Finansal Portföy Stratejisi")
    plt.axhline(y=START_CAPITAL, color='white', ls='--', alpha=0.3)
    plt.title(f"ML + Finansal Backtest V2 | Maks Hisse: {N_STOCKS} | Stop Loss: {STOP_LOSS*100}%", fontsize=14)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f} TL'))
    plt.legend()
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    
    chart_file = "portfolio_backtest_v2_chart.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Portföy grafiği '{chart_file}' dosyasına kaydedildi.")
    plt.show()

if __name__ == "__main__":
    main()
