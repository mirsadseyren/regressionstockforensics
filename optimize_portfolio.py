import pandas as pd
import numpy as np
import os
import multiprocessing
import itertools
import time
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from regression_nonperiodic import (
    load_data, get_tickers_from_file, get_vectorized_metrics,
    STOX_FILE, START_CAPITAL, COMMISSION_RATE
)

import warnings
warnings.filterwarnings('ignore')

# Global variables for worker processes
worker_raw_data = None
worker_daily_candidates = None
worker_trading_days = None

def init_worker(shared_raw_data, shared_daily_candidates, shared_trading_days):
    global worker_raw_data, worker_daily_candidates, worker_trading_days
    worker_raw_data = shared_raw_data
    worker_daily_candidates = shared_daily_candidates
    worker_trading_days = shared_trading_days

def run_simulation_portfolio(params):
    num_stocks, stop_loss, hold_days = params
    try:
        max_hold_delta = pd.Timedelta(days=hold_days)
        current_cash = START_CAPITAL
        active_portfolio = [] # {'t': ticker, 'l': lots, 'b': buy_price, 'max_p': peak_price, 'buy_dt': date, 'days_held': int}
        
        for dt in worker_trading_days:
            # A. Mevcut Portföy Kontrolü (Satışlar)
            for item in active_portfolio[:]:
                current_price = worker_raw_data.at[dt, item['t']]
                if pd.isna(current_price) or current_price <= 0:
                    continue
                    
                should_sell = False
                
                # Zaman Stopu
                if dt - item['buy_dt'] >= max_hold_delta:
                    should_sell = True
                
                # Trailing Stop
                if not should_sell:
                    if current_price > item['max_p']:
                        item['max_p'] = current_price
                    if current_price <= item['max_p'] * (1 - stop_loss):
                        should_sell = True
                        
                if should_sell:
                    revenue = item['l'] * current_price * (1 - COMMISSION_RATE)
                    current_cash += revenue
                    active_portfolio.remove(item)
                else:
                    item['days_held'] += 1

            # B. Yeni Alımlar
            empty_slots = num_stocks - len(active_portfolio)
            if empty_slots > 0 and current_cash > 0:
                day_candidates = worker_daily_candidates.get(dt, [])
                
                if day_candidates:
                    active_tickers = {item['t'] for item in active_portfolio}
                    # Portföyde olmayan en iyi adayları filtrele
                    valid_candidates = [c for c in day_candidates if c['t'] not in active_tickers]
                    top_picks = valid_candidates[:empty_slots]
                    
                    if top_picks:
                        cash_per_slot = current_cash / len(top_picks) # Ya da empty_slots'a böl: current_cash / empty_slots
                        cash_per_slot = current_cash / empty_slots
                        
                        for cand in top_picks:
                            buy_price = cand['price']
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
                                        't': cand['t'],
                                        'l': lots,
                                        'b': buy_price,
                                        'max_p': buy_price,
                                        'buy_dt': dt,
                                        'days_held': 0
                                    })
                                    
        # Simülasyon bitti, son günü değerle
        final_dt = worker_trading_days[-1]
        final_val = current_cash
        for item in active_portfolio:
            cp = worker_raw_data.at[final_dt, item['t']]
            if pd.isna(cp) or cp <= 0:
                cp = item['b']
            final_val += item['l'] * cp
            
        roi = ((final_val - START_CAPITAL) / START_CAPITAL) * 100
        return {
            'num_stocks': num_stocks,
            'stop_loss': stop_loss,
            'hold_days': hold_days,
            'final_balance': final_val,
            'roi': roi
        }
    except Exception as e:
        return {'error': str(e), 'params': params}

def main():
    print("--- 🤖 PORTFOLIO ML GRID SEARCH BAŞLIYOR ---")
    
    # 1. Modeli Eğit ve Hazırla
    matrix_file = 'historical_trade_metrics.csv'
    if not os.path.exists(matrix_file):
        print(f"Hata: {matrix_file} bulunamadı.")
        return

    print("Historical matrix yükleniyor ve KNN modeli eğitiliyor...")
    hist_df = pd.read_csv(matrix_file)
    features = ['slope', 'r2', 'score']
    
    scaler = StandardScaler()
    X_hist_scaled = scaler.fit_transform(hist_df[features])
    
    knn = NearestNeighbors(n_neighbors=50, algorithm='auto')
    knn.fit(X_hist_scaled)
    
    # 2. Verileri Yükle
    print("Borsa verileri yükleniyor...")
    tickers = get_tickers_from_file(STOX_FILE)
    all_data = load_data(tickers)
    
    if isinstance(all_data.columns, pd.MultiIndex):
        try:
            raw_data = all_data['Close'].dropna(axis=1, how='all')
        except KeyError:
            raw_data = all_data.xs('Close', axis=1, level=0).dropna(axis=1, how='all')
    else:
        raw_data = all_data
        
    sim_start_date = (datetime.now() - timedelta(days=365)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if not isinstance(raw_data.index, pd.DatetimeIndex):
        raw_data.index = pd.to_datetime(raw_data.index)
        
    trading_days = raw_data.loc[sim_start_date:].index
    
    print("Teknik metrikler hesaplanıyor...")
    precalc = get_vectorized_metrics(all_data, lookback_days=20)
    
    # 3. Günlük Adayları Önceden Hesapla (Precomputation)
    print("Her gün için KNN tahminleri önceden hesaplanıyor (Bu işlem grid search'ü uçuracak)...")
    daily_candidates = {}
    
    max_candidates_needed = 10 # N=10'a kadar test edeceğiz, ilk 10 yeterli
    
    for dt in tqdm(trading_days, desc="Precomputing Daily Predictions"):
        idx = precalc['prices'].index.get_indexer([dt], method='pad')[0]
        if idx < 0:
            continue
            
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
        
        if not today_df.empty:
            X_today_scaled = scaler.transform(today_df[features])
            distances, indices = knn.kneighbors(X_today_scaled)
            
            # Tüm komşuların kazanç ortalaması ve win rate'i
            # Numpy ile daha hızlı yapılabilir ama zaten precompute olduğu için for döngüsü yeterince hızlı
            expected_pl = np.zeros(len(today_df))
            win_rates = np.zeros(len(today_df))
            
            pl_matrix = hist_df['pl_pct'].values[indices] # Şekil: (num_candidates, 50)
            
            expected_pl = np.mean(pl_matrix, axis=1)
            win_rates = np.mean(pl_matrix > 0, axis=1) * 100
            
            today_df['exp_pl'] = expected_pl
            today_df['win_rate'] = win_rates
            today_df['confidence_score'] = (today_df['win_rate'] / 100) * today_df['exp_pl']
            
            best_cands = today_df[(today_df['exp_pl'] > 0) & (today_df['win_rate'] >= 50)]
            best_cands = best_cands.sort_values(by='confidence_score', ascending=False).head(max_candidates_needed)
            
            # Dictionary formatında sakla
            cands_list = []
            for ticker, row in best_cands.iterrows():
                cands_list.append({
                    't': ticker,
                    'price': row['price']
                })
            daily_candidates[dt] = cands_list
            
    print(f"Precomputation tamamlandı! {len(daily_candidates)} işlem günü için tahminler hazır.")

    # 4. Parametre Uzayı
    param_grid = {
        'num_stocks': [1, 2, 3, 5, 10],
        'stop_loss': [0.03, 0.05, 0.07, 0.10, 0.15],
        'hold_days': [3, 5, 7, 10, 14, 21]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [v for v in itertools.product(*values)]
    
    print(f"\nToplam test edilecek strateji kombinasyonu: {len(combinations)}")
    
    # 5. Multiprocessing Grid Search
    start_time = time.time()
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"{num_workers} çekirdek ile Grid Search başlatılıyor...")
    
    results = []
    
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(raw_data, daily_candidates, trading_days)) as pool:
        pbar = tqdm(total=len(combinations), desc="Grid Search")
        
        for res in pool.imap_unordered(run_simulation_portfolio, combinations, chunksize=1):
            if 'error' not in res:
                results.append(res)
            # else:
            #     print("Error:", res['error'])
            pbar.update(1)
            
        pbar.close()
        
    print(f"Optimizasyon Tamamlandı! Geçen Süre: {time.time() - start_time:.2f} saniye")
    
    # 6. Sonuçları Raporla
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    print("\n" + "="*80)
    print(f"{'RANK':<5} | {'ROI (%)':<10} | {'FINAL BAL':<15} | {'N_STOCKS':<8} | {'STOP_LOSS':<9} | {'HOLD_DAYS':<9}")
    print("-" * 80)
    
    for i, res in enumerate(results[:20]):
        print(f"{i+1:<5} | %{res['roi']:<9.2f} | {res['final_balance']:<15,.2f} | {res['num_stocks']:<8} | {res['stop_loss']:<9.2f} | {res['hold_days']:<9}")

    print("="*80)
    
    if results:
        best = results[0]
        print("\n🏆 EN İYİ STRATEJİ PARAMETRELERİ:")
        print(f"Maksimum Hisse Sayısı (num_stocks) : {best['num_stocks']}")
        print(f"Zarar Kes Oranı (stop_loss)        : %{best['stop_loss']*100:.0f}")
        print(f"Elde Tutma Süresi (hold_days)      : {best['hold_days']} Gün")
        print(f"Beklenen 1 Yıllık Getiri           : %{best['roi']:.2f}")
        print("\nBu parametreleri asıl backtest scriptinizde kullanmak için:")
        print(f"python backtest_portfolio.py -n {best['num_stocks']} -s {best['stop_loss']} -d {best['hold_days']}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
