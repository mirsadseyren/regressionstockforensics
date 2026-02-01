from tqdm import tqdm
import multiprocessing
import itertools
import time
import pandas as pd
import sys
from regression import (
    run_simulation, load_data, get_tickers_from_file, STOX_FILE,
    START_CAPITAL, COMMISSION_RATE, REBALANCE_FREQ
)

# Global variable to hold data in worker processes
worker_data = None

def init_worker(shared_data):
    global worker_data
    worker_data = shared_data

def evaluate_strategy(params):
    """
    Worker function to run a single simulation with given parameters.
    params: (lookback, slope, r2, stop_loss, atr)
    """
    lookback, slope, r2, stop_loss, atr, slope_stop = params
    
    try:
        daily_vals, _, final_balance = run_simulation(
            worker_data,
            lookback_days=lookback,
            min_slope=slope,
            min_r2=r2,
            stop_loss_rate=stop_loss,
            slope_stop_factor=slope_stop,
            max_atr_percent=atr,
            rebalance_freq=REBALANCE_FREQ,
            start_capital=START_CAPITAL,
            commission_rate=COMMISSION_RATE,
            silent=True
        )
        
        roi = ((final_balance - START_CAPITAL) / START_CAPITAL) * 100
        return {
            'lookback': lookback,
            'slope': slope,
            'r2': r2,
            'stop_loss': stop_loss,
            'slope_stop': slope_stop,
            'atr': atr,
            'final_balance': final_balance,
            'roi': roi
        }
    except Exception as e:
        return {
            'error': str(e),
            'params': params
        }

def main():
    print("--- STRATEGY OPTIMIZER (BRUTE FORCE) ---")
    
    # 1. Load Data
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        print("No tickers found.")
        return

    print("Loading data...")
    all_data = load_data(tickers)
    print(f"Data Loaded. Shape: {all_data.shape}")
    
    # 2. Define Parameter Grid
    # Refined grid based on previous best (Lookback ~21, Slope ~0.03, R2 ~0.60)
    param_grid = {
        'lookback': [15, 20, 25, 30],
        'slope': [0.005, 0.01, 0.02, 0.03],
        'r2': [0.50, 0.60, 0.70],
        'stop_loss': [0.05, 0.10, 0.15],
        'atr': [0.06, 0.08, 0.10],
        'slope_stop': [0.0, 0.002, 0.005, 0.01]
    }
    
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    print(f"Total Combinations to Test: {len(combinations)}")
    
    # 3. Run Optimization
    start_time = time.time()
    
    # Use roughly 75% of CPU cores
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Running on {num_workers} processes...")
    
    results = []
    
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(all_data,)) as pool:
        # Map tasks
        # Chunksize adjustment might help performance
        chunk_size = max(1, len(combinations) // (num_workers * 4))
        
        # tqdm progress bar
        pbar = tqdm(total=len(combinations), desc="Optimizing")
        
        for res in pool.imap_unordered(evaluate_strategy, combinations, chunksize=chunk_size):
            if 'error' not in res:
                results.append(res)
            pbar.update(1)
            
        pbar.close()
            
    print("\nOptimization Finished!")
    print(f"Time Elapsed: {time.time() - start_time:.2f} seconds")
    
    # 4. Sort and Display Top 10
    print("\n--- TOP 10 STRATEGIES ---")
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    print(f"{'RANK':<5} {'ROI (%)':<10} {'BALANCE':<15} {'LOOK':<6} {'SLOPE':<8} {'R2':<6} {'STOP':<6} {'S-STOP':<8} {'ATR':<6}")
    print("-" * 95)
    
    for i, res in enumerate(results[:15]):
        print(f"{i+1:<5} {res['roi']:<10.2f} {res['final_balance']:<15,.2f} {res['lookback']:<6} {res['slope']:<8.4f} {res['r2']:<6.2f} {res['stop_loss']:<6.2f} {res['slope_stop']:<8.2f} {res['atr']:<6.2f}")

    # 5. Save best params to a file (Optional)
    if results:
        best = results[0]
        print("\n\nBest Parameters:")
        print(f"Lookback: {best['lookback']}")
        print(f"Slope: {best['slope']}")
        print(f"R2: {best['r2']}")
        print(f"Stop Loss: {best['stop_loss']}")
        print(f"Slope Stop: {best['slope_stop']}")
        print(f"ATR Limit: {best['atr']}")

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()
