import pandas as pd
import numpy as np
import os
import multiprocessing
import time
import pickle
import random
from datetime import datetime, timedelta

from regression_nonperiodic import (
    load_data, get_tickers_from_file,
    STOX_FILE, START_CAPITAL, COMMISSION_RATE
)

import warnings
warnings.filterwarnings('ignore')

# Global variables for worker processes
worker_raw_data = None
worker_daily_candidates = None
worker_trading_days = None
worker_hold_days = None

def init_worker(shared_raw_data, shared_daily_candidates, shared_trading_days, shared_hold_days):
    global worker_raw_data, worker_daily_candidates, worker_trading_days, worker_hold_days
    worker_raw_data = shared_raw_data
    worker_daily_candidates = shared_daily_candidates
    worker_trading_days = shared_trading_days
    worker_hold_days = shared_hold_days

def run_simulation_portfolio(params):
    num_stocks, stop_loss, min_win_rate, min_conf, max_conf = params
    try:
        max_hold_delta = pd.Timedelta(days=worker_hold_days)
        current_cash = START_CAPITAL
        active_portfolio = [] 
        
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
                else:
                    item['days_held'] += 1

            # B. Yeni Alımlar
            empty_slots = num_stocks - len(active_portfolio)
            if empty_slots > 0 and current_cash > 0:
                today_df = worker_daily_candidates.get(dt, pd.DataFrame())
                
                if not today_df.empty:
                    active_tickers = {item['t'] for item in active_portfolio}
                    today_df = today_df.drop(index=list(active_tickers), errors='ignore')
                    
                    # Filtreleme
                    best_candidates = today_df[
                        (today_df['exp_pl'] > 0) & 
                        (today_df['win_rate'] >= min_win_rate) &
                        (today_df['confidence_score'] >= min_conf) &
                        (today_df['confidence_score'] <= max_conf)
                    ]
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
            'min_win_rate': min_win_rate,
            'min_conf': min_conf,
            'max_conf': max_conf,
            'final_balance': final_val,
            'roi': roi,
            'max_dd': max_dd
        }
    except Exception as e:
        return {'error': str(e), 'params': params}

def main():
    print("--- 🤖 SLIDING WINDOW PORTFOLIO GENETİK ARAMA (HEURISTIC) BAŞLIYOR ---")
    
    # 1. Matrisi Yükle
    dump_file = 'sliding_window_predictions.pkl'
    if not os.path.exists(dump_file):
        print(f"Hata: {dump_file} bulunamadı. Lütfen önce backtest_portfolio_fixed.py dosyasını çalıştırın!")
        return

    print("Sliding Window (0 Look-ahead) matrisi belleğe alınıyor...")
    with open(dump_file, "rb") as f:
        saved_data = pickle.load(f)
        
    hold_days = saved_data["hold_days"]
    daily_predictions = saved_data["predictions"]
    print(f"✅ Matris başarıyla yüklendi. (Sabit hold_days: {hold_days})")
    
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

    if not isinstance(raw_data.index, pd.DatetimeIndex):
        raw_data.index = pd.to_datetime(raw_data.index)
        
    trading_days = pd.DatetimeIndex(list(daily_predictions.keys())).sort_values()
    
    if len(trading_days) == 0:
        print("Kayıtlı tahmin bulunamadı!")
        return
        
    print(f"Toplam Simüle Edilecek Gün: {len(trading_days)}")

    # 3. Genetik Algoritma (Heuristic Search) Kurulumu
    bounds = {
        'num_stocks': (1, 8),
        'stop_loss': (0.01, 0.15),
        'min_win_rate': (30.0, 80.0),
        'min_conf': (0.0, 20.0),
        'max_conf': (10.0, 100.0)
    }
    
    def generate_random_individual():
        return (
            random.randint(bounds['num_stocks'][0], bounds['num_stocks'][1]),
            round(random.uniform(bounds['stop_loss'][0], bounds['stop_loss'][1]), 3),
            round(random.uniform(bounds['min_win_rate'][0], bounds['min_win_rate'][1]), 1),
            round(random.uniform(bounds['min_conf'][0], bounds['min_conf'][1]), 2),
            round(random.uniform(bounds['max_conf'][0], bounds['max_conf'][1]), 2)
        )
        
    def mutate(ind):
        ind = list(ind)
        idx = random.randint(0, 4)
        if idx == 0: ind[0] = random.randint(bounds['num_stocks'][0], bounds['num_stocks'][1])
        elif idx == 1: ind[1] = round(random.uniform(bounds['stop_loss'][0], bounds['stop_loss'][1]), 3)
        elif idx == 2: ind[2] = round(random.uniform(bounds['min_win_rate'][0], bounds['min_win_rate'][1]), 1)
        elif idx == 3: ind[3] = round(random.uniform(bounds['min_conf'][0], bounds['min_conf'][1]), 2)
        elif idx == 4: ind[4] = round(random.uniform(max(ind[3]+1, bounds['max_conf'][0]), bounds['max_conf'][1]), 2)
        return tuple(ind)
        
    def crossover(ind1, ind2):
        pt = random.randint(1, 3)
        return ind1[:pt] + ind2[pt:]

    POPULATION_SIZE = 500
    GENERATIONS = 100
    
    print(f"\n🧬 Genetik Algoritma (Heuristic Search) Başlatılıyor...")
    print(f"Popülasyon: {POPULATION_SIZE}, Jenerasyon: {GENERATIONS} (Toplam Test: {POPULATION_SIZE * GENERATIONS})")
    
    population = [generate_random_individual() for _ in range(POPULATION_SIZE)]
    
    start_time = time.time()
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
    
    best_results_dict = {}
    
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(raw_data, daily_predictions, trading_days, hold_days)) as pool:
        for gen in range(GENERATIONS):
            results = list(pool.imap_unordered(run_simulation_portfolio, population))
            valid_results = [r for r in results if 'error' not in r]
            
            def get_fitness(res):
                if res['roi'] <= 0:
                    return res['roi']
                # Max Drawdown cezası
                return res['roi'] * (1 - res['max_dd'])
                
            valid_results.sort(key=get_fitness, reverse=True)
            
            for r in valid_results[:10]:
                param_tuple = (r['num_stocks'], r['stop_loss'], r['min_win_rate'], r['min_conf'], r['max_conf'])
                if param_tuple not in best_results_dict or get_fitness(best_results_dict[param_tuple]) < get_fitness(r):
                    best_results_dict[param_tuple] = r
            
            if valid_results:
                best = valid_results[0]
                best_fitness = get_fitness(best)
                print(f"Gen {gen+1:02d}/{GENERATIONS} | Skor: {best_fitness:<6.0f} | ROI: %{best['roi']:<8.2f} | MaxDD: %{best['max_dd']*100:<5.1f} | N: {best['num_stocks']}, SL: %{best['stop_loss']*100:.1f}, WinR: %{best['min_win_rate']:.1f}, Conf: [{best['min_conf']:.1f}, {best['max_conf']:.1f}]")
            
            # Elitizm: En iyi 15'i doğrudan bir sonraki nesle aktar
            next_gen = [(r['num_stocks'], r['stop_loss'], r['min_win_rate'], r['min_conf'], r['max_conf']) for r in valid_results[:15]]
            
            # Çaprazlama ve Mutasyon (Top 30 havuzundan seçerek)
            parents = valid_results[:30]
            while len(next_gen) < POPULATION_SIZE - 10: 
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                t1 = (p1['num_stocks'], p1['stop_loss'], p1['min_win_rate'], p1['min_conf'], p1['max_conf'])
                t2 = (p2['num_stocks'], p2['stop_loss'], p2['min_win_rate'], p2['min_conf'], p2['max_conf'])
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
    all_best_results.sort(key=lambda x: x['roi'] * (1 - x['max_dd']), reverse=True)
    
    print("\n" + "="*120)
    print(f"{'RANK':<5} | {'SCORE':<8} | {'ROI (%)':<10} | {'MAX DD':<8} | {'FINAL BAL':<15} | {'N_STOCKS':<8} | {'STOP_LOSS':<9} | {'MIN_WIN':<8} | {'MIN_CONF':<8} | {'MAX_CONF':<8}")
    print("-" * 120)
    
    for i, res in enumerate(all_best_results[:20]):
        score = res['roi'] * (1 - res['max_dd'])
        print(f"{i+1:<5} | {score:<8.0f} | %{res['roi']:<9.2f} | %{res['max_dd']*100:<7.2f} | {res['final_balance']:<15,.2f} | {res['num_stocks']:<8} | {res['stop_loss']:<9.2f} | {res['min_win_rate']:<8.1f} | {res['min_conf']:<8.1f} | {res['max_conf']:<8.1f}")

    print("="*120)
    
    if all_best_results:
        best = all_best_results[0]
        print("\n🏆 EN İYİ (SİLİNDİR GİBİ) STRATEJİ PARAMETRELERİ:")
        print(f"Maksimum Hisse Sayısı (num_stocks) : {best['num_stocks']}")
        print(f"Zarar Kes Oranı (stop_loss)        : %{best['stop_loss']*100:.0f}")
        print(f"Elde Tutma Süresi (hold_days)      : {hold_days} Gün (Mühürlü)")
        print(f"Min Kazanma İhtimali (win_rate)    : %{best['min_win_rate']}")
        print(f"ML Güven Skoru Aralığı             : [{best['min_conf']}, {best['max_conf']}]")
        print(f"Beklenen Getiri (ROI)              : %{best['roi']:.2f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
