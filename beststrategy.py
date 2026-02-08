from tqdm import tqdm
import multiprocessing
import itertools
import time
import pandas as pd
import sys
import os

# Add endeksler to path if needed for import
sys.path.append(os.path.join(os.path.dirname(__file__), 'endeksler'))
from endeks1y import get_top_indices_tickers

from regression_nonperiodic import (
    run_simulation, load_data, get_tickers_from_file, STOX_FILE,
    START_CAPITAL, COMMISSION_RATE, REBALANCE_FREQ
)

# Global variables to hold data in worker processes
worker_data = None
worker_ticker_map = None

def init_worker(shared_data, shared_ticker_map):
    global worker_data, worker_ticker_map
    worker_data = shared_data
    worker_ticker_map = shared_ticker_map

def evaluate_strategy(params):
    """
    Worker function to run a single simulation with given parameters.
    params: (lookback, slope, r2, stop_loss, atr, slope_stop, rebalance, num_indices)
    """
    lookback, slope, r2, stop_loss, atr, slope_stop, rebalance, num_indices = params
    
    try:
        # Filter data for specific tickers based on num_indices
        target_tickers = worker_ticker_map.get(num_indices, [])
        if not target_tickers:
            return {
                'error': f'No tickers for num_indices={num_indices}',
                'params': params
            }
            
        # Select columns efficiently
        # Assuming worker_data is a DataFrame or MultiIndex DataFrame
        # We need to filter the columns that are in target_tickers
        if isinstance(worker_data.columns, pd.MultiIndex):
            # level 1 is ticker
            # Keep columns where level 1 matches target_tickers
            # Creating a mask or using loc can be tricky with MultiIndex slices if not sorted
            # Easier approach: intersection
            valid_tickers = [t for t in target_tickers if t in worker_data.columns.levels[1]]
            if not valid_tickers:
                 return {'roi': -999, 'final_balance': 0, 'params': params}
            
            # Using loc with slice(None) for first level (Open, Close...) and list of tickers for second
            filtered_data = worker_data.loc[:, (slice(None), valid_tickers)]
        else:
            # Simple index
            valid_tickers = [t for t in target_tickers if t in worker_data.columns]
            if not valid_tickers:
                 return {'roi': -999, 'final_balance': 0, 'params': params}
            filtered_data = worker_data[valid_tickers]

        daily_vals, _, final_balance = run_simulation(
            filtered_data,
            lookback_days=lookback,
            min_slope=slope,
            min_r2=r2,
            stop_loss_rate=stop_loss,
            slope_stop_factor=slope_stop,
            max_atr_percent=atr,
            rebalance_freq=rebalance,
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
            'rebalance': rebalance,
            'num_indices': num_indices,
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
    
    # 1. Pre-calculate indices mappings
    print("Fetching index tickers map (1-10)...")
    ticker_map = {}
    all_needed_tickers = set()
    
    for i in range(1, 11):
        t_list = get_top_indices_tickers(i)
        ticker_map[i] = t_list
        all_needed_tickers.update(t_list)
        print(f"Top {i} indices -> {len(t_list)} tickers")
        
    if not all_needed_tickers:
        print("Error: Could not fetch tickers from endeks1y.")
        return

    # 2. Load Data for ALL involved tickers
    print(f"Loading data for {len(all_needed_tickers)} total unique tickers...")
    all_data = load_data(list(all_needed_tickers))
    print(f"Data Loaded. Shape: {all_data.shape}")
    
    # 3. Define Parameter Grid
    # Refined grid based on previous best
    param_grid = {
        'lookback': [15, 20, 25, 30],
        'slope': [0.02, 0.025, 0.03], # Reduced for speed as we added another dimension
        'r2': [0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        'stop_loss': [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        'atr': [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        'slope_stop': [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05],
        'rebalance': ['3D', '7D', '15D', '30D', '60D'],
        'num_indices': list(range(1, 21)) # 1 to 10
    }
    
    # Adjust grid size as needed. Currently 2*1*1*1*1*2*2*10 = 80 combinations. Very fast.
    # Let's expand a bit if it's too small.
    param_grid['lookback'] = [15, 20, 25]
    param_grid['slope'] = [0.01, 0.02, 0.03]
    
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    print(f"Total Combinations to Test: {len(combinations)}")
    
    # 4. Run Optimization
    start_time = time.time()
    
    # Use roughly 75% of CPU cores
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Running on {num_workers} processes...")
    
    results = []
    
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(all_data, ticker_map)) as pool:
        # Map tasks
        chunk_size = max(1, len(combinations) // (num_workers * 4))
        
        # tqdm progress bar
        pbar = tqdm(total=len(combinations), desc="Optimizing")
        
        for res in pool.imap_unordered(evaluate_strategy, combinations, chunksize=chunk_size):
            if 'error' not in res:
                results.append(res)
            # else:
            #    print(res['error'])
            pbar.update(1)
            
        pbar.close()
            
    print("\nOptimization Finished!")
    print(f"Time Elapsed: {time.time() - start_time:.2f} seconds")
    
    # 5. Sort and Display Top 10
    print("\n--- TOP 10 STRATEGIES ---")
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    print(f"{'RANK':<5} {'ROI (%)':<10} {'BALANCE':<15} {'IDX':<4} {'LOOK':<6} {'SLOPE':<8} {'R2':<6} {'STOP':<6} {'S-STOP':<8} {'ATR':<6} {'REBAL':<6}")
    print("-" * 110)
    
    for i, res in enumerate(results[:20]):
        print(f"{i+1:<5} {res['roi']:<10.2f} {res['final_balance']:<15,.2f} {res['num_indices']:<4} {res['lookback']:<6} {res['slope']:<8.4f} {res['r2']:<6.2f} {res['stop_loss']:<6.2f} {res['slope_stop']:<8.2f} {res['atr']:<6.2f} {res['rebalance']:<6}")

    # 6. Save best params to a file (Optional)
    if results:
        best = results[0]
        print("\n\nBest Parameters:")
        print(f"Num Indices: {best['num_indices']}")
        print(f"Lookback: {best['lookback']}")
        print(f"Slope: {best['slope']}")
        print(f"R2: {best['r2']}")
        print(f"Stop Loss: {best['stop_loss']}")
        print(f"Slope Stop: {best['slope_stop']}")
        print(f"ATR Limit: {best['atr']}")
        print(f"Rebalance Freq: {best['rebalance']}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
