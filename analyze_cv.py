import pandas as pd
import numpy as np
import re
from regression_nonperiodic import run_simulation, load_data, get_tickers_from_file, STOX_FILE, LOOKBACK_DAYS

def calculate_cv(prices):
    """
    Calculates the Coefficient of Variation (CV) for a given series of prices.
    CV = Standard Deviation / Mean
    """
    if len(prices) < 2:
        return 0.0
    return prices.std() / prices.mean()

def parse_and_analyze_cv(trade_history, all_data):
    """
    Parses trade history to identify profitable trades, retrieves the price data 
    used for the buy signal (lookback window), and calculates the CV.
    """
    
    # Regex to extract P/L percentage
    sell_pattern = re.compile(r"P/L:\s*%([-\d.]+)")
    
    # Track when a stock was bought: {ticker: buy_date_obj}
    # Note: trade_history dates are strings 'YYYY-MM-DD'
    open_positions = {}
    
    # Pre-process data for easier access
    if isinstance(all_data.columns, pd.MultiIndex):
        closes = all_data['Close'].dropna(axis=1, how='all')
    else:
        closes = all_data
    
    # Ensure index is datetime for proper slicing
    if not isinstance(closes.index, pd.DatetimeIndex):
        closes.index = pd.to_datetime(closes.index)

    # Regex for Buy Info: (Eğim: 0.0384, R2: 0.94)
    buy_pattern = re.compile(r"Eğim:\s*([\d.]+),\s*R2:\s*([\d.]+)")

    print("\n" + "="*120)
    print(f"{'DATE':<12} {'TICKER':<10} {'PROFIT %':<10} {'R2':<8} {'SLOPE':<10} {'CV (Lookback)':<15} {'INFO':<40}")
    print("="*120)
    
    successful_trades_count = 0
    total_cv = 0
    cv_values = []
    profitable_data = []
    
    for row in trade_history:
        # row: [Date, Ticker, Lots, Price, Action, Cash, Info]
        if len(row) < 7: continue
        
        date_str = row[0]
        ticker = row[1]
        action = row[4]
        info = row[6]
        
        db_ticker = ticker
        if db_ticker not in closes.columns:
            db_ticker = ticker + ".IS"
        if db_ticker not in closes.columns:
            pass
            
        if action == "ALIS":
            # Extract R2 and Slope
            r2, slope = 0.0, 0.0
            match = buy_pattern.search(info)
            if match:
                slope = float(match.group(1))
                r2 = float(match.group(2))
            
            open_positions[ticker] = {
                'date': date_str,
                'r2': r2,
                'slope': slope
            }
            
        elif action in ["SATIS", "STOP LOSS", "SLOPE STOP", "SÜRE DOLDU (7D)"] or "P/L" in info:
            match = sell_pattern.search(info)
            if match:
                try:
                    pl_pct = float(match.group(1))
                    
                    buy_info = open_positions.pop(ticker, None)
                    
                    if pl_pct > 0 and buy_info:
                        buy_date_str = buy_info['date']
                        
                        # 1. Identify the lookback window for this trade
                        buy_date = pd.to_datetime(buy_date_str)
                        
                        try:
                            idx = closes.index.get_indexer([buy_date], method='pad')[0]
                            
                            if idx >= LOOKBACK_DAYS:
                                start_idx = idx - LOOKBACK_DAYS
                                window_data = closes[db_ticker].iloc[start_idx : idx + 1]
                                
                                cv = calculate_cv(window_data)
                                
                                print(f"{date_str:<12} {ticker:<10} %{pl_pct:<9.2f} {buy_info['r2']:<8.4f} {buy_info['slope']:<10.6f} {cv:<15.6f} {info:<40}")
                                successful_trades_count += 1
                                total_cv += cv
                                cv_values.append(cv)
                                
                                profitable_data.append({
                                    'pl': pl_pct,
                                    'r2': buy_info['r2'],
                                    'slope': buy_info['slope']
                                })
                        except Exception as e:
                            pass
                            
                except ValueError:
                    continue
                    
    print("="*120)
    
    if successful_trades_count > 0:
        avg_cv = total_cv / successful_trades_count
        print(f"Total Profitable Trades: {successful_trades_count}")
        print(f"Average CV of Winners: {avg_cv:.6f}")
        
        if len(cv_values) > 1:
            print(f"Min CV: {min(cv_values):.6f}")
            print(f"Max CV: {max(cv_values):.6f}")
            
        # P/L Weighted Averages
        if profitable_data:
            df = pd.DataFrame(profitable_data)
            weights = df['pl']
            
            # Weighted Average Formula: Sum(Val * Weight) / Sum(Weights)
            w_avg_r2 = np.average(df['r2'], weights=weights)
            w_avg_slope = np.average(df['slope'], weights=weights)
            
            # Weighted Standard Deviation
            # Formula: sqrt( sum( w * (x - weighted_mean)^2 ) / sum(w) )
            variance_r2 = np.average((df['r2'] - w_avg_r2)**2, weights=weights)
            w_std_r2 = np.sqrt(variance_r2)
            
            variance_slope = np.average((df['slope'] - w_avg_slope)**2, weights=weights)
            w_std_slope = np.sqrt(variance_slope)
            
            # Suggested Thresholds (Various Levels)
            # Loose: Mean - 1.0 * StdDev
            loose_r2 = w_avg_r2 - w_std_r2
            loose_slope = w_avg_slope - w_std_slope
            
            # Moderate: Mean - 0.5 * StdDev
            mod_r2 = w_avg_r2 - (0.5 * w_std_r2)
            mod_slope = w_avg_slope - (0.5 * w_std_slope)
            
            # Strict: Mean - 0.25 * StdDev
            strict_r2 = w_avg_r2 - (0.25 * w_std_r2)
            strict_slope = w_avg_slope - (0.25 * w_std_slope)
            
            print("-" * 80)
            print("STATISTICAL OPTIMIZATION (Weighted by Profit %):")
            print(f"{'Metric':<10} {'W.Mean':<10} {'W.StdDev':<10} {'Loose (-1.0s)':<15} {'Mod (-0.5s)':<15} {'Strict (-0.25s)':<15}")
            print("-" * 80)
            print(f"{'R2':<10} {w_avg_r2:<10.4f} {w_std_r2:<10.4f} {loose_r2:<15.4f} {mod_r2:<15.4f} {strict_r2:<15.4f}")
            print(f"{'Slope':<10} {w_avg_slope:<10.6f} {w_std_slope:<10.6f} {loose_slope:<15.6f} {mod_slope:<15.6f} {strict_slope:<15.6f}")
            print("-" * 80)
            print("Note: Choose 'Moderate' or 'Strict' if 'Loose' lets in too many false positives.")
    
    print("="*120)

if __name__ == "__main__":
    print("Loading data...")
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        print("No tickers found.")
        exit()
        
    all_data = load_data(tickers)
    print("Data loaded.\n")
    
    print("Running simulation...")
    daily_vals, trade_history, final_balance = run_simulation(all_data, silent=False)
    
    print("\nAnalyzing CV values for profitable trades...")
    parse_and_analyze_cv(trade_history, all_data)
