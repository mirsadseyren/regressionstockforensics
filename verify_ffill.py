import pandas as pd
import numpy as np
from regression import get_vectorized_metrics, load_data, get_tickers_from_file, STOX_FILE

tickers = get_tickers_from_file(STOX_FILE)
data = load_data(tickers)

# Test vectorized metrics with ffill
precalc = get_vectorized_metrics(data, 30)

# Check if ffill is working: find a ticker that has NaNs at the end of the original data
if isinstance(data.columns, pd.MultiIndex):
    closes = data['Close']
else:
    closes = data

# Find tickers with NaN in the last row but data in the previous rows
last_row = closes.iloc[-1]
nan_tickers = last_row[last_row.isna()].index
if not nan_tickers.empty:
    chosen_ticker = nan_tickers[0]
    print(f"Checking ticker with terminal NaN: {chosen_ticker}")
    
    # In precalc, this ticker should now have a valid (ffilled) price and slope
    price_val = precalc['prices'].at[precalc['prices'].index[-1], chosen_ticker]
    slope_val = precalc['slopes'].at[precalc['slopes'].index[-1], chosen_ticker]
    
    print(f"Original price last row: {closes.at[closes.index[-1], chosen_ticker]}")
    print(f"Precalc price last row (ffilled): {price_val}")
    print(f"Precalc slope last row (ffilled): {slope_val}")
    
    if not np.isnan(price_val):
        print("SUCCESS: Forward fill is working for terminal NaNs.")
    else:
        print("NOTE: Forward fill did not catch this (maybe more than 5 days of NaNs).")
else:
    print("All tickers have data in the last row. Cannot verify ffill easily with current cache.")

# Check last date discovery logic
valid_dates = closes.index[closes.notna().any(axis=1)]
print(f"Data index end: {closes.index[-1]}")
print(f"Last valid date found: {valid_dates[-1] if not valid_dates.empty else 'None'}")
