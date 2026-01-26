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

def fix_cache(df=None):
    if df is None:
        if not os.path.exists(DATA_CACHE_FILE):
            print(f"Error: {DATA_CACHE_FILE} not found.")
            return None

        print("Loading cache from file...")
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
        if 'Close' in df.columns.get_level_values(0):
            tickers = df.columns.get_level_values(1).unique()
            
            for ticker in tickers:
                try:
                    close_series = df[('Close', ticker)]
                    
                    vals = close_series.values
                    ratios = np.zeros(len(vals))
                    ratios[:] = 1.0 # Default no change
                    
                    mask_valid = (vals[:-1] != 0) & (~np.isnan(vals[:-1])) & (~np.isnan(vals[1:]))
                    curr_ratios = np.zeros(len(vals)-1)
                    curr_ratios[mask_valid] = vals[1:][mask_valid] / vals[:-1][mask_valid]
                    curr_ratios[~mask_valid] = 1.0
                    
                    drop_indices = np.where(curr_ratios < 0.6)[0] + 1
                    
                    if len(drop_indices) > 0:
                        print(f"Adjusting {ticker}...")
                        cum_adj_factor = np.ones(len(vals))
                        for idx in drop_indices:
                             factor = curr_ratios[idx-1]
                             print(f"  - Split at {df.index[idx].date()}: Factor {factor:.4f}")
                             cum_adj_factor[:idx] *= factor
                        
                        for col_attr in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                            if (col_attr, ticker) in df.columns:
                                df.loc[:, (col_attr, ticker)] *= cum_adj_factor
                                
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
        else:
             print("Unknown MultiIndex structure. Attempting to adjust all columns independently.")
             for col in df.columns:
                 df[col] = adjust_prior_prices(df[col])

    else:
        print("Single Index detected (Assuming Close prices). Adjusting each column...")
        for col in df.columns:
            df[col] = adjust_prior_prices(df[col])

    # Save to file only if we didn't receive an input dataframe (standalone mode)
    # Actually, we might want to save it anyway if it's the main cache.
    # But for flexibility, let's just return it and let the caller decide.
    # If run from CLI, we save it.
    return df

if __name__ == "__main__":
    fixed_df = fix_cache()
    if fixed_df is not None:
        print("Saving fixed cache...")
        fixed_df.to_pickle(DATA_CACHE_FILE)
        print("Done.")

