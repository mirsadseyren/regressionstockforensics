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
import yfinance as yf

import random

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
    num_stocks, stop_loss, hold_days, min_conf, max_conf, min_fin, max_fin = params
    try:
        max_hold_delta = pd.Timedelta(days=hold_days)
        current_cash = START_CAPITAL
        active_portfolio = [] # {'t': ticker, 'l': lots, 'b': buy_price, 'max_p': peak_price, 'buy_dt': date}
        
        peak_portfolio_val = START_CAPITAL
        max_dd = 0.0
        
        for dt in worker_trading_days:
            # 1. Portföy Değerini Hesapla ve Max DD'yi Güncelle
            daily_val = current_cash
            for item in active_portfolio:
                cp = worker_raw_data.at[dt, item['t']]
                if pd.isna(cp) or cp <= 0:
                    cp = item['b']
                daily_val += item['l'] * cp
                
            if daily_val > peak_portfolio_val:
                peak_portfolio_val = daily_val
                
            dd = (peak_portfolio_val - daily_val) / peak_portfolio_val
            if dd > max_dd:
                max_dd = dd

            # 2. Mevcut Portföy Kontrolü (Satışlar)
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

            # B. Yeni Alımlar
            empty_slots = num_stocks - len(active_portfolio)
            if empty_slots > 0 and current_cash > 0:
                day_candidates = worker_daily_candidates.get(dt, [])
                
                if day_candidates:
                    active_tickers = {item['t'] for item in active_portfolio}
                    # Portföyde olmayan ve thresholdları geçen adayları filtrele
                    valid_candidates = [
                        c for c in day_candidates 
                        if c['t'] not in active_tickers and
                           c['confidence_score'] >= min_conf and
                           c['confidence_score'] <= max_conf and
                           c['finansal_skor'] >= min_fin and
                           c['finansal_skor'] <= max_fin
                    ]
                    top_picks = valid_candidates[:empty_slots]
                    
                    if top_picks:
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
                                        'buy_dt': dt
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
            'min_conf': min_conf,
            'max_conf': max_conf,
            'min_fin': min_fin,
            'max_fin': max_fin,
            'final_balance': final_val,
            'roi': roi,
            'max_dd': max_dd
        }
    except Exception as e:
        return {'error': str(e), 'params': params}

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
    print("--- 🤖 PORTFOLIO V2 ML GRID SEARCH BAŞLIYOR ---")
    
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

    # 2.1 Finansal Skoru Hesapla
    financial_scores_dict = fetch_and_calculate_financial_score(raw_data.columns)
        
    sim_start_date = (datetime.now() - timedelta(days=365)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if not isinstance(raw_data.index, pd.DatetimeIndex):
        raw_data.index = pd.to_datetime(raw_data.index)
        
    trading_days = raw_data.loc[sim_start_date:].index
    
    print("Teknik metrikler hesaplanıyor...")
    precalc = get_vectorized_metrics(all_data, lookback_days=20)
    
    # 3. Günlük Adayları Önceden Hesapla (Precomputation)
    print("Her gün için KNN tahminleri önceden hesaplanıyor (Bu işlem grid search'ü uçuracak)...")
    daily_candidates = {}
    
    # Max threshold limitini kaldırıyoruz, tüm listeyi precompute etmeli ki backtest_v2 ile aynı çalışsın.
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
            
            pl_matrix = hist_df['pl_pct'].values[indices] # Şekil: (num_candidates, 50)
            
            expected_pl = np.mean(pl_matrix, axis=1)
            win_rates = np.mean(pl_matrix > 0, axis=1) * 100
            
            today_df['exp_pl'] = expected_pl
            today_df['win_rate'] = win_rates
            today_df['confidence_score'] = (today_df['win_rate'] / 100) * today_df['exp_pl']
            
            # --- FİNANSAL SKOR ENTEGRASYONU ---
            fin_scores = [financial_scores_dict.get(t, 0.0) for t in today_df.index]
            today_df['finansal_skor'] = fin_scores
            today_df['fin_x_kume_guveni'] = today_df['finansal_skor'] * today_df['confidence_score']
            
            best_cands = today_df[(today_df['exp_pl'] > 0) & (today_df['win_rate'] >= 50)]
            # Yeni sıralama kriteri ile (Finansal x Küme) sırala
            best_cands = best_cands.sort_values(by='fin_x_kume_guveni', ascending=False)
            
            # Dictionary formatında sakla
            cands_list = []
            for ticker, row in best_cands.iterrows():
                cands_list.append({
                    't': ticker,
                    'price': row['price'],
                    'confidence_score': row['confidence_score'],
                    'finansal_skor': row['finansal_skor']
                })
            daily_candidates[dt] = cands_list
            
    print(f"Precomputation tamamlandı! {len(daily_candidates)} işlem günü için tahminler hazır.")


    # 4. Genetik Algoritma (Heuristic Search) Kurulumu
    bounds = {
        'num_stocks': (1, 5),
        'stop_loss': (0.01, 0.15),
        'hold_days': (3, 21),
        'min_conf': (0.0, 20.0),
        'max_conf': (10.0, 100.0),
        'min_fin': (-10.0, 15.0),
        'max_fin': (10.0, 100.0)
    }
    
    def generate_random_individual():
        return (
            random.randint(bounds['num_stocks'][0], bounds['num_stocks'][1]),
            round(random.uniform(bounds['stop_loss'][0], bounds['stop_loss'][1]), 3),
            random.randint(bounds['hold_days'][0], bounds['hold_days'][1]),
            round(random.uniform(bounds['min_conf'][0], bounds['min_conf'][1]), 2),
            round(random.uniform(bounds['max_conf'][0], bounds['max_conf'][1]), 2),
            round(random.uniform(bounds['min_fin'][0], bounds['min_fin'][1]), 2),
            round(random.uniform(bounds['max_fin'][0], bounds['max_fin'][1]), 2)
        )
        
    def mutate(ind):
        ind = list(ind)
        idx = random.randint(0, 6)
        if idx == 0: ind[0] = random.randint(bounds['num_stocks'][0], bounds['num_stocks'][1])
        elif idx == 1: ind[1] = round(random.uniform(bounds['stop_loss'][0], bounds['stop_loss'][1]), 3)
        elif idx == 2: ind[2] = random.randint(bounds['hold_days'][0], bounds['hold_days'][1])
        elif idx == 3: ind[3] = round(random.uniform(bounds['min_conf'][0], bounds['min_conf'][1]), 2)
        elif idx == 4: ind[4] = round(random.uniform(max(ind[3]+1, bounds['max_conf'][0]), bounds['max_conf'][1]), 2)
        elif idx == 5: ind[5] = round(random.uniform(bounds['min_fin'][0], bounds['min_fin'][1]), 2)
        elif idx == 6: ind[6] = round(random.uniform(max(ind[5]+1, bounds['max_fin'][0]), bounds['max_fin'][1]), 2)
        return tuple(ind)
        
    def crossover(ind1, ind2):
        pt = random.randint(1, 5)
        return ind1[:pt] + ind2[pt:]

    POPULATION_SIZE = 1000
    GENERATIONS = 200
    
    print(f"\n🧬 Genetik Algoritma (Heuristic Search) Başlatılıyor...")
    print(f"Popülasyon: {POPULATION_SIZE}, Jenerasyon: {GENERATIONS} (Toplam Test: {POPULATION_SIZE * GENERATIONS})")
    print(f"Büyük arama uzayında hayatta kalma ve mutasyon ile en iyi genler aranıyor...")
    
    population = [generate_random_individual() for _ in range(POPULATION_SIZE)]
    
    start_time = time.time()
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    best_results_dict = {}
    
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(raw_data, daily_candidates, trading_days)) as pool:
        for gen in range(GENERATIONS):
            results = list(pool.imap_unordered(run_simulation_portfolio, population))
            valid_results = [r for r in results if 'error' not in r]
            
            def get_fitness(res):
                if res['roi'] <= 0:
                    return res['roi']
                return res['roi'] * (1 - res['max_dd'])
                
            valid_results.sort(key=get_fitness, reverse=True)
            
            for r in valid_results[:10]:
                param_tuple = (r['num_stocks'], r['stop_loss'], r['hold_days'], r['min_conf'], r['max_conf'], r['min_fin'], r['max_fin'])
                if param_tuple not in best_results_dict or get_fitness(best_results_dict[param_tuple]) < get_fitness(r):
                    best_results_dict[param_tuple] = r
            
            best = valid_results[0]
            best_fitness = get_fitness(best)
            print(f"Gen {gen+1:02d}/{GENERATIONS} | Skor: {best_fitness:<6.0f} | ROI: %{best['roi']:<8.2f} | MaxDD: %{best['max_dd']*100:<5.1f} | N: {best['num_stocks']}, SL: %{best['stop_loss']*100:.1f}, Hold: {best['hold_days']}, Conf: [{best['min_conf']:.1f}, {best['max_conf']:.1f}], Fin: [{best['min_fin']:.1f}, {best['max_fin']:.1f}]")
            
            # Elitizm: En iyi 20'yi doğrudan bir sonraki nesle aktar
            next_gen = [(r['num_stocks'], r['stop_loss'], r['hold_days'], r['min_conf'], r['max_conf'], r['min_fin'], r['max_fin']) for r in valid_results[:20]]
            
            # Çaprazlama ve Mutasyon (Top 40 havuzundan seçerek)
            parents = valid_results[:40]
            while len(next_gen) < POPULATION_SIZE - 15: # Kalan boşluğa çaprazlama
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                t1 = (p1['num_stocks'], p1['stop_loss'], p1['hold_days'], p1['min_conf'], p1['max_conf'], p1['min_fin'], p1['max_fin'])
                t2 = (p2['num_stocks'], p2['stop_loss'], p2['hold_days'], p2['min_conf'], p2['max_conf'], p2['min_fin'], p2['max_fin'])
                child = crossover(t1, t2)
                if random.random() < 0.4:  # %40 mutasyon ihtimali
                    child = mutate(child)
                next_gen.append(child)
                
            # Rastgele keşif: Yeni kanlar ekle
            while len(next_gen) < POPULATION_SIZE:
                next_gen.append(generate_random_individual())
                
            population = next_gen
            
    print(f"\nOptimizasyon Tamamlandı! Geçen Süre: {time.time() - start_time:.2f} saniye")
    
    # 6. Sonuçları Raporla
    all_best_results = list(best_results_dict.values())
    # En iyi skorlar (ROI ve Max DD birleşik fitness)
    all_best_results.sort(key=lambda x: x['roi'] * (1 - x['max_dd']), reverse=True)
    
    print("\n" + "="*135)
    print(f"{'RANK':<5} | {'SCORE':<8} | {'ROI (%)':<10} | {'MAX DD':<8} | {'FINAL BAL':<15} | {'N_STOCKS':<8} | {'STOP_LOSS':<9} | {'HOLD_DAYS':<9} | {'MIN_CONF':<8} | {'MAX_CONF':<8} | {'MIN_FIN':<8} | {'MAX_FIN':<8}")
    print("-" * 135)
    
    for i, res in enumerate(all_best_results[:20]):
        score = res['roi'] * (1 - res['max_dd'])
        print(f"{i+1:<5} | {score:<8.0f} | %{res['roi']:<9.2f} | %{res['max_dd']*100:<7.2f} | {res['final_balance']:<15,.2f} | {res['num_stocks']:<8} | {res['stop_loss']:<9.2f} | {res['hold_days']:<9} | {res['min_conf']:<8.1f} | {res['max_conf']:<8.1f} | {res['min_fin']:<8.1f} | {res['max_fin']:<8.1f}")

    print("="*120)
    
    if all_best_results:
        best = all_best_results[0]
        print("\n🏆 EN İYİ STRATEJİ PARAMETRELERİ (V2 Heuristic):")
        print(f"Maksimum Hisse Sayısı (num_stocks) : {best['num_stocks']}")
        print(f"Zarar Kes Oranı (stop_loss)        : %{best['stop_loss']*100:.0f}")
        print(f"Elde Tutma Süresi (hold_days)      : {best['hold_days']} Gün")
        print(f"ML Güven Skoru Aralığı             : [{best['min_conf']}, {best['max_conf']}]")
        print(f"Finansal Güven Skoru Aralığı       : [{best['min_fin']}, {best['max_fin']}]")
        print(f"Beklenen 1 Yıllık Getiri           : %{best['roi']:.2f}")
        print("\nBu parametreleri asıl backtest v2 scriptinizde kullanmak için:")
        print(f"python portfolio_backtest_v2.py -n {best['num_stocks']} -s {best['stop_loss']} -d {best['hold_days']} --min-conf {best['min_conf']} --max-conf {best['max_conf']} --min-fin {best['min_fin']} --max-fin {best['max_fin']}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
