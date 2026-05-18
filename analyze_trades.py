import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

# Import functions from our main file
from regression_nonperiodic import (
    load_data, get_tickers_from_file, run_simulation,
    STOX_FILE, START_CAPITAL, COMMISSION_RATE
)

warnings.filterwarnings('ignore')

def main():
    print("--- TRADES ANALYSIS & K-MEANS CLUSTERING ---")
    
    # 1. Load Data
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        print("No tickers found!")
        return
        
    print(f"Loading data for {len(tickers)} tickers...")
    all_data = load_data(tickers)
    print("Data loaded successfully.")
    
    # 2. Gather ALL possible trades using Vectorized Metrics
    print("\nRunning vectorized analysis to gather ALL possible trade signals (No portfolio constraints)...")
    from regression_nonperiodic import get_vectorized_metrics
    from tqdm import tqdm
    
    lookback_days = 20
    hold_days = 7 # We will check the return after 7 days
    
    print("Calculating technical metrics for all dates...")
    precalc = get_vectorized_metrics(all_data, lookback_days)
    
    prices = precalc['prices']
    future_prices = prices.shift(-hold_days)
    pl_pct_matrix = (future_prices - prices) / prices * 100
    
    slopes = precalc['slopes']
    r2 = precalc['r2']
    discounts = precalc['discounts']
    
    # Mask to find reasonable signals (very relaxed to gather data)
    # We ignore negative slopes and completely random data (r2 < 0.2)
    mask = (slopes > 0.0) & (r2 > 0.2) & (prices > 0)
    
    print("Extracting features and calculating future returns...")
    valid_slopes = slopes[mask].stack()
    valid_r2 = r2[mask].stack()
    valid_score = discounts[mask].stack()
    valid_pl = pl_pct_matrix[mask].stack()
    
    # Combine into a DataFrame
    metrics_df = pd.DataFrame({
        'slope': valid_slopes,
        'r2': valid_r2,
        'score': valid_score,
        'pl_pct': valid_pl
    }).dropna() # Drop NAs (like the last 7 days where future price is unknown)
    
    metrics_collector = metrics_df.to_dict('records')
    
    if not metrics_collector:
        print("No trades were made! Try relaxing parameters even more.")
        return
        
    print(f"\nCollected data for {len(metrics_collector)} total potential trades!")
    
    # 3. Prepare Data for Clustering
    df = pd.DataFrame(metrics_collector)
    
    # Features for clustering
    X = df[['slope', 'r2', 'score']].copy()
    
    # Scale features since they have different ranges
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # We want to find clusters using DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=20) # You can adjust eps and min_samples to tune the clustering sensitivity
    df['cluster'] = dbscan.fit_predict(X_scaled)
    
    total_trades = len(df)
    
    # 4. Analyze Clusters
    print("\n--- DBSCAN CLUSTER ANALYSIS ---")
    cluster_stats = df.groupby('cluster').agg(
        trade_count=('pl_pct', 'count'),
        avg_pl=('pl_pct', 'mean'),
        win_rate=('pl_pct', lambda x: (x > 0).mean() * 100),
        avg_slope=('slope', 'mean'),
        avg_r2=('r2', 'mean'),
        avg_score=('score', 'mean')
    ).reset_index()
    
    # Calculate Opportunity % (Alım Fırsatı Yüzdesi)
    cluster_stats['opp_pct'] = (cluster_stats['trade_count'] / total_trades) * 100
    
    # Sort clusters by Average P/L descending
    cluster_stats = cluster_stats.sort_values(by='avg_pl', ascending=False)
    
    print(f"{'Cluster':<8} | {'Count':<6} | {'Opp %':<7} | {'Avg P/L %':<10} | {'Win Rate %':<11} | {'Avg Slope':<10} | {'Avg R2':<8} | {'Avg Score (Dist)'}")
    print("-" * 100)
    for _, row in cluster_stats.iterrows():
        c_name = f"C-{int(row['cluster'])}" if row['cluster'] != -1 else "Noise(-1)"
        print(f"{c_name:<8} | {int(row['trade_count']):<6} | {row['opp_pct']:<7.2f} | {row['avg_pl']:<10.2f} | {row['win_rate']:<11.2f} | {row['avg_slope']:<10.4f} | {row['avg_r2']:<8.2f} | {row['avg_score']:<8.4f}")
        
    valid_clusters = cluster_stats[cluster_stats['cluster'] != -1]
    if not valid_clusters.empty:
        best_cluster_data = valid_clusters.iloc[0]
        best_cluster_id = best_cluster_data['cluster']
    else:
        best_cluster_data = cluster_stats.iloc[0]
        best_cluster_id = best_cluster_data['cluster']
    
    print("\n💡 OPTIMAL PARAMETER RECOMMENDATION (Based on best cluster):")
    print(f"Recommended Min Slope: ~ {best_cluster_data['avg_slope']:.4f}")
    print(f"Recommended Min R2: ~ {best_cluster_data['avg_r2']:.2f}")
    print(f"Recommended Target Distance (Score): ~ {best_cluster_data['avg_score']:.4f}")
    
    # Save the huge matrix to a file for future predictions
    print("\nSaving the massive 3D matrix to 'historical_trade_metrics.csv'...")
    df.to_csv('historical_trade_metrics.csv', index=False)
    
    # 5. Scatter Plot Visualization (3D)
    print("\nPreparing 3D Scatter Plot...")
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define color map based on P/L %
    # Profitable trades = Green, Losing trades = Red
    colors = ['lime' if p > 0 else 'red' for p in df['pl_pct']]
    sizes = [abs(p) + 10 for p in df['pl_pct']] # Larger dots for bigger wins/losses
    
    scatter = ax.scatter(
        df['slope'], 
        df['r2'], 
        df['score'], 
        c=colors, 
        s=sizes, 
        alpha=0.6,
        edgecolors='w',
        linewidth=0.5
    )
    
    # Plot cluster centers (calculated as mean of points in cluster)
    first_x_plotted = False
    for i, row in valid_clusters.iterrows():
        cid = row['cluster']
        center = [row['avg_slope'], row['avg_r2'], row['avg_score']]
        if cid == best_cluster_id:
            ax.scatter(center[0], center[1], center[2], c='cyan', marker='*', s=500, label='Best Cluster Center')
        else:
            ax.scatter(center[0], center[1], center[2], c='magenta', marker='X', s=200, label='Other Centers' if not first_x_plotted else "")
            first_x_plotted = True

    ax.set_title('Trade Metrics 3D Scatter (Green=Profit, Red=Loss)', fontsize=14)
    ax.set_xlabel('Slope (Eğim)')
    ax.set_ylabel('R-Squared (R²)')
    ax.set_zlabel('Score (Distance from Reg)')
    
    # Create legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Profitable', markerfacecolor='lime', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Loss', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Best Cluster', markerfacecolor='cyan', markersize=15),
        Line2D([0], [0], marker='X', color='w', label='Other Clusters', markerfacecolor='magenta', markersize=10)
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
