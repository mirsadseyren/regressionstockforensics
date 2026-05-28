import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

from regression_nonperiodic import (
    load_data, get_tickers_from_file, get_vectorized_metrics, STOX_FILE
)

def main():
    parser = argparse.ArgumentParser(description="Test Trades Simulation")
    parser.add_argument('-d', '--hold-days', type=int, default=7, help="Elde tutma süresi (gün olarak, varsayılan: 7)")
    parser.add_argument('-n', '--top-n', type=int, default=1, help="Her gün seçilecek hisse sayısı (varsayılan: 1)")
    args = parser.parse_args()

    HOLD_DAYS = args.hold_days

    print(f"--- 📊 TEST TRADES ({HOLD_DAYS} Günlük Beklenti vs Gerçekleşme) ---")
    
    # 1. Market Verilerini Yükle
    print("Borsa verileri yükleniyor...")
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

    print("Teknik metrikler vektörize olarak hesaplanıyor...")
    precalc = get_vectorized_metrics(all_data, lookback_days=20)
    
    # 2. Geçmiş Veri Havuzunu Tarihli Olarak Oluştur (Look-ahead bias önlemek için)
    print("Tarih bazlı geçmiş işlem havuzu oluşturuluyor...")
    prices = precalc['prices']
    # İleriye dönük P/L (sadece model eğitimi havuzu için, gün gün filtrelenecek)
    future_prices = prices.shift(-HOLD_DAYS)
    pl_pct_matrix = (future_prices - prices) / prices * 100
    
    slopes = precalc['slopes']
    r2 = precalc['r2']
    discounts = precalc['discounts']
    
    # Temel filtreler (analyze_trades.py ile aynı)
    mask = (slopes > 0.0) & (r2 > 0.2) & (prices > 0)
    
    metrics_df = pd.DataFrame({
        'slope': slopes[mask].stack(),
        'r2': r2[mask].stack(),
        'score': discounts[mask].stack(),
        'pl_pct': pl_pct_matrix[mask].stack()
    }).dropna().reset_index()
    
    # Eğer multi-index'in isimleri yoksa varsayılan olarak level_0 ve level_1 olur
    metrics_df.rename(columns={metrics_df.columns[0]: 'Date', metrics_df.columns[1]: 'Ticker'}, inplace=True)

    # 3. 1 Yıllık simülasyon başlangıcı
    sim_start_date = (datetime.now() - timedelta(days=365)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    trading_days = raw_data.loc[sim_start_date:].index
    
    if len(trading_days) == 0:
        print("Geçerli işlem günü bulunamadı!")
        return

    print("Teknik metrikler vektörize olarak hesaplanıyor...")
    precalc = get_vectorized_metrics(all_data, lookback_days=20)

    # 3. Gün gün iterasyon
    HOLD_DAYS = args.hold_days # Komut satırından gelen gün sayısı
    
    results = []
    
    total_days = len(trading_days)
    # Gelecekteki 7. günü bulabilmek için iterasyonu biraz erken bitiriyoruz veya shift ile buluyoruz.
    
    for i, dt in tqdm(enumerate(trading_days), total=total_days, desc="Daily Test"):
        idx = precalc['prices'].index.get_indexer([dt], method='pad')[0]
        if idx < 0:
            continue
            
        # O gün için KNN tahminleri
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
        
        # Filtreler (backtest_portfolio.py ile aynı)
        today_df = today_df[(today_df['slope'] > 0) & (today_df['r2'] > 0.1) & (today_df['price'] > 0)]
        
        # Look-ahead bias engelleme: Sadece sonucu bilinen geçmiş işlemleri kullan
        past_idx = max(0, idx - HOLD_DAYS)
        cutoff_date = precalc['prices'].index[past_idx]
        
        hist_df = metrics_df[metrics_df['Date'] <= cutoff_date]
        features = ['slope', 'r2', 'score']
        
        K = 50
        if not today_df.empty and len(hist_df) >= K:
            # Sadece bugüne kadar sonucu belli olmuş işlemleri ölçekle ve KNN eğit
            scaler = StandardScaler()
            X_hist_scaled = scaler.fit_transform(hist_df[features])
            
            knn = NearestNeighbors(n_neighbors=K, algorithm='auto')
            knn.fit(X_hist_scaled)

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
            today_df['confidence_score'] = (today_df['win_rate'] / 100) * today_df['exp_pl']
            
            best_candidates = today_df[(today_df['exp_pl'] > 0) & (today_df['win_rate'] >= 50)]
            best_candidates = best_candidates.sort_values(by='confidence_score', ascending=False)
            
            top_pick = best_candidates.head(args.top_n)
            
            if not top_pick.empty:
                for ticker in top_pick.index:
                    buy_price = top_pick.loc[ticker, 'price']
                    exp_pl = top_pick.loc[ticker, 'exp_pl']
                    conf_score = top_pick.loc[ticker, 'confidence_score']
                    
                    # HOLD_DAYS gün sonrasının fiyatını bul
                    sell_idx = idx + HOLD_DAYS
                    
                    # Eğer simülasyonun sonlarına geldiysek ve o gün sonrası henüz yoksa
                    if sell_idx < len(precalc['prices']):
                        sell_date = precalc['prices'].index[sell_idx]
                        sell_price = precalc['prices'].iloc[sell_idx][ticker]
                        
                        if not pd.isna(sell_price) and sell_price > 0:
                            actual_pl = ((sell_price - buy_price) / buy_price) * 100
                            
                            results.append({
                                'Date': dt.strftime('%Y-%m-%d'),
                                'Sell Date': sell_date.strftime('%Y-%m-%d'),
                                'Ticker': ticker,
                                'Buy Price': buy_price,
                                'Sell Price': sell_price,
                                'Confidence Score': conf_score,
                                'Expected P/L (%)': exp_pl,
                                'Actual P/L (%)': actual_pl,
                                'Met Expectation': actual_pl >= exp_pl,
                                'Is Profit': actual_pl > 0
                            })

    # 4. İstatistikleri Hesapla ve Yazdır
    if not results:
        print("\nUygun işlem bulunamadı.")
        return

    results_df = pd.DataFrame(results)
    
    total_trades = len(results_df)
    win_trades = results_df['Is Profit'].sum()
    win_rate = (win_trades / total_trades) * 100
    
    avg_exp_pl = results_df['Expected P/L (%)'].mean()
    avg_act_pl = results_df['Actual P/L (%)'].mean()
    
    met_exp_count = results_df['Met Expectation'].sum()
    met_exp_rate = (met_exp_count / total_trades) * 100
    
    # Beklenti sapması (Actual - Expected)
    results_df['Deviation'] = results_df['Actual P/L (%)'] - results_df['Expected P/L (%)']
    avg_deviation = results_df['Deviation'].mean()

    print("\n" + "="*50)
    print(f"📈 TEST TRADES İSTATİSTİKLERİ ({args.hold_days} İşlem Günlük Elde Tutma)")
    print("="*50)
    print(f"Toplam İşlem Günü (Sinyal Alınan): {total_trades}")
    print(f"Başarı Oranı (Kârla Kapanan)    : %{win_rate:.2f}")
    print(f"Ortalama Beklenen Getiri (Exp)  : %{avg_exp_pl:.2f}")
    print(f"Ortalama Gerçekleşen Getiri(Act): %{avg_act_pl:.2f}")
    print(f"Beklentiyi Karşılama Oranı      : %{met_exp_rate:.2f} (Gerçekleşen >= Beklenen)")
    print(f"Ortalama Sapma (Act - Exp)      : %{avg_deviation:.2f}")
    print("="*50)
    
    print("\nEn İyi Performans Gösteren 5 İşlem:")
    top5 = results_df.sort_values(by='Actual P/L (%)', ascending=False).head(5)
    for _, row in top5.iterrows():
        print(f"{row['Date']} | {row['Ticker']:<6} | Beklenen: %{row['Expected P/L (%)']:>5.2f} | Gerçekleşen: %{row['Actual P/L (%)']:>5.2f}")
        
    print("\nEn Kötü Performans Gösteren 5 İşlem:")
    bot5 = results_df.sort_values(by='Actual P/L (%)', ascending=True).head(5)
    for _, row in bot5.iterrows():
        print(f"{row['Date']} | {row['Ticker']:<6} | Beklenen: %{row['Expected P/L (%)']:>5.2f} | Gerçekleşen: %{row['Actual P/L (%)']:>5.2f}")

    # 5. Verileri Yuvarla ve Excel/CSV olarak kaydet
    results_df['Buy Price'] = results_df['Buy Price'].round(2)
    results_df['Sell Price'] = results_df['Sell Price'].round(2)
    results_df['Expected P/L (%)'] = results_df['Expected P/L (%)'].round(2)
    results_df['Actual P/L (%)'] = results_df['Actual P/L (%)'].round(2)
    results_df['Confidence Score'] = results_df['Confidence Score'].round(4)
    results_df['Deviation'] = results_df['Deviation'].round(2)

    out_file_csv = 'test_trades_results.csv'
    results_df.to_csv(out_file_csv, index=False, sep=';', decimal=',')
    print(f"\nDetaylı sonuçlar '{out_file_csv}' dosyasına kaydedildi.")
    
    # Excel formatında da kaydetmeyi deneyelim (Eğer openpyxl kuruluysa)
    try:
        import openpyxl
        out_file_xlsx = 'test_trades_results.xlsx'
        results_df.to_excel(out_file_xlsx, index=False)
        print(f"Ayrıca renkli ve düzgün formatlı olarak '{out_file_xlsx}' dosyasına da kaydedildi.")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
