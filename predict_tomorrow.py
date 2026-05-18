import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

# Import our existing logic
from regression_nonperiodic import (
    load_data, get_tickers_from_file, get_vectorized_metrics,
    STOX_FILE
)

warnings.filterwarnings('ignore')

def main():
    print("--- 🤖 MACHINE LEARNING PREDICTION FOR TOMORROW ---")
    
    # 1. Load Historical Matrix
    matrix_file = 'historical_trade_metrics.csv'
    if not os.path.exists(matrix_file):
        print(f"Error: {matrix_file} not found! Please run 'analyze_trades.py' first.")
        return
        
    print("Loading historical trade matrix...")
    hist_df = pd.read_csv(matrix_file)
    print(f"Loaded {len(hist_df):,} historical trades.")
    
    # We use slope, r2, and score as our 3D coordinates
    features = ['slope', 'r2', 'score']
    
    # Scale historical data
    scaler = StandardScaler()
    X_hist_scaled = scaler.fit_transform(hist_df[features])
    
    # Fit K-Nearest Neighbors to find similar historical situations
    # We will look at the 50 most similar past setups to determine reliability
    K = 50
    print(f"Fitting KNN model (K={K})...")
    knn = NearestNeighbors(n_neighbors=K, algorithm='auto')
    knn.fit(X_hist_scaled)
    
    # 2. Get Current Market Data
    print("\nFetching latest market data...")
    tickers = get_tickers_from_file(STOX_FILE)
    all_data = load_data(tickers)
    
    # 3. Calculate Metrics
    print("Calculating current technical indicators...")
    precalc = get_vectorized_metrics(all_data, lookback_days=20)
    
    target_date_str = input("\nHangi tarih için tahmin yapmak istersiniz? (Örn: 2024-05-15) [Boş bırakırsanız BUGÜN/YARIN için çalışır]: ").strip()
    
    if target_date_str:
        target_date = pd.to_datetime(target_date_str)
        valid_dates = precalc['prices'].index[precalc['prices'].index <= target_date]
        if len(valid_dates) == 0:
            print("Verilen tarihten önce borsa verisi bulunamadı!")
            return
        dt = valid_dates[-1]
        print(f"Seçilen/En yakın işlem tarihi: {dt.strftime('%Y-%m-%d')}")
        idx = precalc['prices'].index.get_loc(dt)
    else:
        dt = precalc['prices'].index[-1]
        print(f"En son veri tarihi kullanılıyor: {dt.strftime('%Y-%m-%d')}")
        idx = -1
        
    slopes = precalc['slopes'].iloc[idx]
    r2 = precalc['r2'].iloc[idx]
    discounts = precalc['discounts'].iloc[idx]
    prices = precalc['prices'].iloc[idx]
    
    actual_pl_dict = {}
    is_past_date = (idx != -1 and idx != len(precalc['prices']) - 1)
    if is_past_date:
        # If it's a past date, check the actual price 7 trading days later to see if model was right!
        future_idx = idx + 7
        if future_idx < len(precalc['prices']):
            future_prices = precalc['prices'].iloc[future_idx]
            actual_pl_dict = ((future_prices - prices) / prices * 100).to_dict()
    
    # Create DataFrame for candidates
    today_df = pd.DataFrame({
        'slope': slopes,
        'r2': r2,
        'score': discounts,
        'price': prices
    })
    
    # Basic filters to remove complete garbage
    today_df = today_df[(today_df['slope'] > 0) & (today_df['r2'] > 0.1) & (today_df['price'] > 0)]
    
    if today_df.empty:
        print("No valid candidates found today even with relaxed filters.")
        return
        
    # 4. Predict Expected Profit and Win Rate using KNN
    print(f"Analyzing {len(today_df)} valid candidates using historical similarity...")
    X_today_scaled = scaler.transform(today_df[features])
    
    # Find nearest neighbors
    distances, indices = knn.kneighbors(X_today_scaled)
    
    expected_pl = []
    win_rates = []
    
    for i in range(len(today_df)):
        # Get the historical indices for this specific stock's current situation
        neighbor_indices = indices[i]
        neighbor_trades = hist_df.iloc[neighbor_indices]
        
        # Calculate stats from these neighbors
        avg_pl = neighbor_trades['pl_pct'].mean()
        win_rate = (neighbor_trades['pl_pct'] > 0).mean() * 100
        
        expected_pl.append(avg_pl)
        win_rates.append(win_rate)
        
    today_df['exp_pl'] = expected_pl
    today_df['win_rate'] = win_rates
    
    # We want high win rate AND good expected profit.
    # Score them: (Win Rate / 100) * Expected P/L
    today_df['confidence_score'] = (today_df['win_rate'] / 100) * today_df['exp_pl']
    
    # Filter out candidates with negative expected P/L or poor win rates
    best_candidates = today_df[(today_df['exp_pl'] > 0) & (today_df['win_rate'] >= 50)]
    
    if best_candidates.empty:
        print("\n⚠️ WARNING: The AI model could not find any reliable setups for tomorrow. Cash might be king!")
        
        # Show the "least bad" ones anyway
        print("\nHere are the top 5 'least bad' setups:")
        best_candidates = today_df.sort_values(by='confidence_score', ascending=False).head(5)
    else:
        # Sort by confidence score
        best_candidates = best_candidates.sort_values(by='confidence_score', ascending=False)
        print(f"\n✅ Found {len(best_candidates)} highly reliable setups!")
        print("Top 15 Recommendations for Tomorrow:")
        
    print("\n" + "="*100)
    if is_past_date and actual_pl_dict:
        print(f"{'TICKER':<10} | {'PRICE':<8} | {'WIN RATE':<10} | {'EXP. PROFIT':<12} | {'ACTUAL 7D %':<12} | {'SLOPE':<8} | {'R2':<6} | {'SCORE'}")
        print("-" * 100)
        for ticker, row in best_candidates.head(15).iterrows():
            t_name = ticker.replace('.IS', '')
            actual_pl = actual_pl_dict.get(ticker, np.nan)
            actual_str = f"{actual_pl:>8.2f}%" if not pd.isna(actual_pl) else "N/A"
            print(f"{t_name:<10} | {row['price']:<8.2f} | {row['win_rate']:<9.1f}% | {row['exp_pl']:>7.2f}%     | {actual_str:<12} | {row['slope']:<8.4f} | {row['r2']:<6.2f} | {row['score']:<8.4f}")
    else:
        print(f"{'TICKER':<10} | {'PRICE':<8} | {'WIN RATE':<10} | {'EXP. PROFIT':<12} | {'SLOPE':<8} | {'R2':<6} | {'SCORE'}")
        print("-" * 100)
        for ticker, row in best_candidates.head(15).iterrows():
            t_name = ticker.replace('.IS', '')
            print(f"{t_name:<10} | {row['price']:<8.2f} | {row['win_rate']:<9.1f}% | {row['exp_pl']:>7.2f}%     | {row['slope']:<8.4f} | {row['r2']:<6.2f} | {row['score']:<8.4f}")
    print("="*100)
    
if __name__ == "__main__":
    main()
