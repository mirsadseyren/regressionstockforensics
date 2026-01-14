import pandas as pd
import os
from datetime import datetime
from regression import load_data, get_clean_data, get_tickers_from_file, STOX_FILE

tickers = get_tickers_from_file(STOX_FILE)
data = load_data(tickers)
clean_data = get_clean_data(data)

print(f"Original shape: {data.shape}")
print(f"Cleaned shape: {clean_data.shape}")

if isinstance(data.columns, pd.MultiIndex):
    orig_last_date = data.index[-1]
    clean_last_date = clean_data.index[-1]
    print(f"Original last date: {orig_last_date}")
    print(f"Cleaned last date: {clean_last_date}")
    
    last_row = data['Close'].iloc[-1]
    nan_count = last_row.isna().sum()
    print(f"NaNs in original last row: {nan_count} out of {len(last_row)}")
else:
    print("Data is not MultiIndex (unexpected but possible if single ticker).")

if clean_data.shape[0] < data.shape[0]:
    print("SUCCESS: get_clean_data removed terminal NaN rows.")
else:
    print("NOTE: No rows removed. Last row probably had enough data.")
