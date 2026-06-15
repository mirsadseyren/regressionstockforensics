import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

warnings.filterwarnings('ignore')

def main():
    print("--- ML METRİKLERİ KORELASYON ANALİZİ ---")
    
    # 1. Load data
    print("Geçmiş işlem havuzu yükleniyor (historical_trade_metrics.csv)...")
    try:
        df = pd.read_csv('historical_trade_metrics.csv')
    except FileNotFoundError:
        print("Hata: historical_trade_metrics.csv bulunamadı. Lütfen önce analyze_trades.py çalıştırın.")
        return

    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['slope', 'r2', 'score', 'pl_pct'])
    
    # To save time, if dataset is too large, we can sample, but 50k is very fast for KNN
    print(f"Toplam geçerli satır: {len(df)}")
    
    # 2. Extract features for KNN
    features = ['slope', 'r2', 'score']
    X = df[features].values
    y = df['pl_pct'].values
    
    print("Özellikler ölçeklendiriliyor...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. K-Nearest Neighbors to simulate "Beklenen Kar", "Kazanma İhtimali", "Güven Skoru"
    print("K-En Yakın Komşu (KNN) modeli ile ML skorları hesaplanıyor...")
    K = 50
    # Use n_neighbors = K + 1 because the point itself will be found as the nearest neighbor
    knn = NearestNeighbors(n_neighbors=K+1, algorithm='auto', n_jobs=-1)
    knn.fit(X_scaled)
    
    distances, indices = knn.kneighbors(X_scaled)
    
    # Exclude the first index (which is the point itself)
    neighbor_indices = indices[:, 1:]
    
    # Fetch the pl_pct of neighbors
    neighbor_pls = y[neighbor_indices]
    
    print("Beklenen Kar, Kazanma İhtimali ve ML Güven Skoru vektörize olarak hesaplanıyor...")
    exp_pls = np.mean(neighbor_pls, axis=1)
    win_rates = np.mean(neighbor_pls > 0, axis=1) * 100
    confidence_scores = (win_rates / 100) * exp_pls
    
    # Add to dataframe
    df['Beklenen 7G Kar (%)'] = exp_pls
    df['Kazanma İhtimali (%)'] = win_rates
    df['ML Güven Skoru'] = confidence_scores
    
    # Rename original columns for better plotting
    df.rename(columns={
        'pl_pct': '7 Günlük Kar (%)',
        'slope': 'Regresyon Eğimi',
        'r2': 'R2',
        'score': 'Uzaklık Skoru (Score)'
    }, inplace=True)
    
    # Target variable
    target = '7 Günlük Kar (%)'
    metrics = [
        'Regresyon Eğimi',
        'R2',
        'Uzaklık Skoru (Score)',
        'Beklenen 7G Kar (%)',
        'Kazanma İhtimali (%)',
        'ML Güven Skoru'
    ]
    
    # --- NORMALIZATION STEP ---
    print("\nVeriler araştırma için normalize ediliyor (Z-Score Standardization)...")
    normalized_df = df.copy()
    cols_to_normalize = metrics + [target]
    
    for col in cols_to_normalize:
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        
        if std_val != 0 and not pd.isna(std_val):
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
            
    # --- CORRELATION MATRIX ---
    print(f"\n--- Correlations with {target} (Normalized Data) ---")
    corr_matrix = normalized_df[cols_to_normalize].corr()
    
    # Print the correlations with the target
    target_corrs = corr_matrix[target].sort_values(ascending=False)
    for index, value in target_corrs.items():
        if index != target:
            print(f"Correlation {index} vs {target}: {value:.4f}")
            
    # Draw correlation heatmap
    print("\nKorelasyon ısı haritası oluşturuluyor...")
    os.makedirs('correlation_plots/ml_metrics', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('ML Metrikleri ve Özellikleri Korelasyon Matrisi', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_plots/ml_metrics/correlation_heatmap.png')
    plt.close()
    
    # --- SCATTER PLOTS ---
    print("\nScatter grafikleri oluşturuluyor (normalized)...")
    
    for metric in metrics:
        temp_df = normalized_df.dropna(subset=[metric, target]).copy()
        
        # Remove extreme outliers for better visualization on normalized scale (top/bottom 1%)
        if not temp_df.empty and len(temp_df) > 10:
            q_low = temp_df[metric].quantile(0.01)
            q_hi  = temp_df[metric].quantile(0.99)
            q_low_target = temp_df[target].quantile(0.01)
            q_hi_target = temp_df[target].quantile(0.99)
            
            temp_df = temp_df[(temp_df[metric] <= q_hi) & (temp_df[metric] >= q_low) & 
                              (temp_df[target] <= q_hi_target) & (temp_df[target] >= q_low_target)]
        
        # Sample for plotting if too large (scatterplot gets messy with 50k dots)
        if len(temp_df) > 5000:
            plot_df = temp_df.sample(5000, random_state=42)
        else:
            plot_df = temp_df
            
        if not plot_df.empty and len(plot_df) > 1:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x=metric, y=target, alpha=0.3, color='dodgerblue')
            
            # Add trendline
            sns.regplot(data=plot_df, x=metric, y=target, scatter=False, color='red', line_kws={"linewidth":2})
            
            corr = temp_df[metric].corr(temp_df[target]) # Use all data for corr value
            plt.title(f'NORMALIZED: {metric} vs {target}\nCorr: {corr:.3f}')
            plt.xlabel(f"{metric} (Z-Score)")
            plt.ylabel(f"{target} (Z-Score)")
            
            plt.tight_layout()
            safe_metric_name = metric.split('(')[0].strip().replace('%', 'Pct').replace('/', '_').replace(' ', '_').replace('İ', 'I').replace('ı', 'i').replace('ğ', 'g')
            filename = f"correlation_plots/ml_metrics/{safe_metric_name}_vs_Kar_normalized_scatter.png"
            plt.savefig(filename)
            plt.close()
            print(f"Kaydedildi: {filename}")
            
    # Extra: Kazanma Ihtimali vs ML Guven Skoru
    temp_df = normalized_df[['Kazanma İhtimali (%)', 'ML Güven Skoru']].dropna()
    if len(temp_df) > 5000:
        plot_df = temp_df.sample(5000, random_state=42)
    else:
        plot_df = temp_df
        
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x='Kazanma İhtimali (%)', y='ML Güven Skoru', alpha=0.3, color='purple')
    sns.regplot(data=plot_df, x='Kazanma İhtimali (%)', y='ML Güven Skoru', scatter=False, color='red')
    corr = temp_df['Kazanma İhtimali (%)'].corr(temp_df['ML Güven Skoru'])
    plt.title(f'NORMALIZED: Kazanma İhtimali (%) vs ML Güven Skoru\nCorr: {corr:.3f}')
    plt.xlabel('Kazanma İhtimali (%) (Z-Score)')
    plt.ylabel('ML Güven Skoru (Z-Score)')
    plt.tight_layout()
    plt.savefig("correlation_plots/ml_metrics/Kazanma_Ihtimali_vs_Guven_Skoru.png")
    plt.close()
    
    # Save the dataset
    df.to_csv('correlation_plots/ml_metrics/ml_metrics_correlation_data_raw.csv', index=False, sep=';', decimal=',')
    normalized_df.to_csv('correlation_plots/ml_metrics/ml_metrics_correlation_data_normalized.csv', index=False, sep=';', decimal=',')
    print("\nİşlem Tamam! Tüm grafikler ve veri setleri 'correlation_plots/ml_metrics' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
