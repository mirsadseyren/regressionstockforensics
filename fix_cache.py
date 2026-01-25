import pandas as pd
import numpy as np
import os

DATA_CACHE_FILE = 'bist_data_cache.pkl'

def adjust_prior_prices(series, threshold=0.6):
    """
    Detects drops larger than (1-threshold)% (e.g., 40%) and adjusts prior prices.
    This is a backward adjustment.
    """
    # Work on a copy to avoid SettingWithCopy warnings on the original df slices
    series = series.copy()
    
    # We Iterate backwards to handle multiple splits correctly if they exist,
    # though forward iteration with cumulative product is also valid.
    # Let's simple iterate forward to find splits, apply adjustment, then continue.
    # Actually, calculating returns and finding outliers is vectorizable but 
    # applying cumulative adjustment is safer row-by-row or using cumprod.
    
    # Calculate daily returns: price / prev_price
    # We use values to avoid index alignment issues during calculation
    values = series.values
    if len(values) < 2:
        return series

    # Ratios: values[i] / values[i-1]
    # invalid/NaN handling: fill with 1.0
    ratios = values[1:] / values[:-1]
    
    # Identify indices where drop is significant (val < 0.6 * prev_val)
    # This corresponds to index i+1 in the original series
    split_indices = np.where(ratios < threshold)[0] + 1
    
    if len(split_indices) > 0:
        print(f"  Found {len(split_indices)} potential split/rights issue(s).")
        
        # We need to construct a cumulative adjustment factor array
        adj_factors = np.ones(len(series))
        
        for idx in split_indices:
            ratio = values[idx] / values[idx-1]
            if np.isnan(ratio) or ratio == 0:
                 continue
            # All prices BEFORE this index need to be multiplied by this ratio
            print(f"    - Drop detected at index {idx} ({series.index[idx].date()}): Ratio {ratio:.4f}")
            adj_factors[:idx] *= ratio
            
        # Apply adjustment
        series = series * adj_factors
        
    return series

def fix_cache():
    if not os.path.exists(DATA_CACHE_FILE):
        print(f"Error: {DATA_CACHE_FILE} not found.")
        return

    print("Loading cache...")
    df = pd.read_pickle(DATA_CACHE_FILE)
    print(f"Original shape: {df.shape}")

    # 1. Fill gaps (Resample to Business Days and Fill)
    print("Resampling to Business Days ('B') and filling gaps...")
    # Handle duplicates if any
    df = df[~df.index.duplicated(keep='first')]
    
    # Resample and ffill
    df = df.asfreq('B').ffill()
    
    print(f"Shape after resampling: {df.shape}")

    # 2. Adjust for price drops (Splits/Rights Issues)
    print("Checking for significant price drops (>40%)...")
    
    # Check if MultiIndex columns (e.g. yfinance format: Level0=Close/Open, Level1=Ticker)
    is_multiindex = isinstance(df.columns, pd.MultiIndex)
    
    if is_multiindex:
        # Assuming typical yfinance structure where columns are (PriceType, Ticker)
        # We need to identify tickers.
        # df.columns.levels[1] should be tickers if level 0 is attributes
        # Or check if 'Close' is in level 0.
        
        if 'Close' in df.columns.get_level_values(0):
            # Level 0 is Attribute (Close, Open, etc.), Level 1 is Ticker
            tickers = df.columns.get_level_values(1).unique()
            
            for ticker in tickers:
                try:
                    # adjusting Close is critical. We should usually adjust Open, High, Low too.
                    # Let's calculate factors based on Close and apply to OHLC
                    
                    close_series = df[('Close', ticker)]
                    
                    # We'll reuse the logic to find factors, but we need to apply them to O,H,L,C
                    # Let's extract the "adjust_prior_prices" logic to just get factors preferably,
                    # but for now let's just run the adjustment on Close and verify.
                    # Ideally, if Close drops 50%, Open/High/Low should also be adjusted.
                    
                    # Custom logic for whole group adjustment:
                    # Calculate factors from Close
                    vals = close_series.values
                    ratios = np.zeros(len(vals))
                    ratios[:] = 1.0 # Default no change
                    
                    mask_valid = (vals[:-1] != 0) & (~np.isnan(vals[:-1])) & (~np.isnan(vals[1:]))
                    # effective ratios
                    curr_ratios = np.zeros(len(vals)-1)
                    curr_ratios[mask_valid] = vals[1:][mask_valid] / vals[:-1][mask_valid]
                    curr_ratios[~mask_valid] = 1.0
                    
                    # Threshold check
                    drop_indices = np.where(curr_ratios < 0.6)[0] + 1
                    
                    if len(drop_indices) > 0:
                        print(f"Adjusting {ticker}...")
                        cum_adj_factor = np.ones(len(vals))
                        for idx in drop_indices:
                             factor = curr_ratios[idx-1]
                             print(f"  - Split at {df.index[idx].date()}: Factor {factor:.4f}")
                             cum_adj_factor[:idx] *= factor
                        
                        # Apply to all relevant columns for this ticker
                        for col_attr in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                            if (col_attr, ticker) in df.columns:
                                df.loc[:, (col_attr, ticker)] *= cum_adj_factor
                                
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    
        else:
             print("Unknown MultiIndex structure. Attempting to adjust all columns independently (risky if mix of price and vol).")
             # Fallback: treat every column as a price series
             for col in df.columns:
                 df[col] = adjust_prior_prices(df[col])

    else:
        # Single Index Columns - Assume they are tickers and values are Close prices
        print("Single Index detected (Assuming Close prices). Adjusting each column...")
        for col in df.columns:
            # Skip if it looks like Volume (heuristic: mean > 1000000? or name?)
            # Assuming these are stats/prices.
            df[col] = adjust_prior_prices(df[col])

    print("Saving fixed cache...")
    df.to_pickle(DATA_CACHE_FILE)
    print("Done.")

if __name__ == "__main__":
    fix_cache()
