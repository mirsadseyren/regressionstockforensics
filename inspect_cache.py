import pandas as pd
import os

CACHE_FILE = 'bist_data_cache.pkl'

if os.path.exists(CACHE_FILE):
    try:
        data = pd.read_pickle(CACHE_FILE)
        print(f"Data shape: {data.shape}")
        if isinstance(data.index, pd.DatetimeIndex):
            print(f"Start date: {data.index.min()}")
            print(f"End date: {data.index.max()}")
        elif isinstance(data.columns, pd.MultiIndex):
             # For MultiIndex (yfinance format)
             print(f"Start date: {data.index.min()}")
             print(f"End date: {data.index.max()}")
        
        # Check for 2026 data specifically
        data_2026 = data.loc['2026']
        if not data_2026.empty:
            print(f"Found {len(data_2026)} data points in 2026.")
            print(f"Last data point in 2026: {data_2026.index.max()}")
            
            # Check for NaNs in 'Close'
            if isinstance(data.columns, pd.MultiIndex):
                closes_2026 = data_2026['Close']
                nan_counts = closes_2026.isna().sum()
                completely_nan = nan_counts[nan_counts == len(data_2026)]
                print(f"Tickers with 100% NaN in 2026: {len(completely_nan)} out of {len(closes_2026.columns)}")
                if len(completely_nan) > 0:
                    print(f"Example NaN tickers: {completely_nan.index[:5].tolist()}")
                
                valid_tickers = nan_counts[nan_counts < len(data_2026)]
                print(f"Tickers with at least some data in 2026: {len(valid_tickers)}")
        else:
            print("No data found for 2026.")
            
    except Exception as e:
        print(f"Error reading cache: {e}")
else:
    print("Cache file not found.")
