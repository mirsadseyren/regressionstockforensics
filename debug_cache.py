import pandas as pd
import os
from datetime import datetime

DATA_CACHE_FILE = 'bist_data_cache.pkl'

if os.path.exists(DATA_CACHE_FILE):
    data = pd.read_pickle(DATA_CACHE_FILE)
    print(f"Cache index type: {type(data.index)}")
    print(f"Last few dates in cache:\n{data.index[-5:]}")
    
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
        
    print("\nLast 5 rows of Close for a few tickers:")
    tickers = closes.columns[:5]
    print(closes[tickers].tail())
    
    print("\nNumber of non-NaN values in the last row:")
    print(closes.iloc[-1].notna().sum())
    
    mtime = datetime.fromtimestamp(os.path.getmtime(DATA_CACHE_FILE))
    print(f"\nCache file mtime: {mtime}")
else:
    print("Cache file not found.")
