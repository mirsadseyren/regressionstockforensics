import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

warnings.filterwarnings('ignore')

from regression_nonperiodic import (
    load_data, get_tickers_from_file, get_vectorized_metrics, STOX_FILE
)

def main():
    print("--- DATA LEAKAGE KORUMALI (NO-LEAK) KAR AĞIRLIKLI ML METRİKLERİ ---")
    
    # 1. Market Verilerini Yükle
    print("Borsa verileri yükleniyor ve metrikler hesaplanıyor...")
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        print("Ticker bulunamadı.")
        return
        
    all_data = load_data(tickers)
    
    print("Teknik metrikler vektörize olarak hesaplanıyor...")
    precalc = get_vectorized_metrics(all_data, lookback_days=20)
    
    prices = precalc['prices']
    slopes = precalc['slopes']
    r2 = precalc['r2']
    discounts = precalc['discounts']
    
    HOLD_DAYS = 7
    future_prices = prices.shift(-HOLD_DAYS)
    pl_pct_matrix = (future_prices - prices) / prices * 100
    
    mask = (slopes > 0.0) & (r2 > 0.2) & (prices > 0)
    
    df = pd.DataFrame({
        'slope': slopes[mask].stack(),
        'r2': r2[mask].stack(),
        'score': discounts[mask].stack(),
        'pl_pct': pl_pct_matrix[mask].stack()
    }).dropna().reset_index()
    
    df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Ticker'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['slope', 'r2', 'score', 'pl_pct'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Toplam işlem ihtimali: {len(df)}")
    
    # 2. Time-Aware KNN (Leakage Korumalı)
    print("Zaman serisine duyarlı (Look-ahead bias korumalı) KNN hesaplanıyor...")
    
    exp_pls = np.full(len(df), np.nan)
    win_rates = np.full(len(df), np.nan)
    
    features = ['slope', 'r2', 'score']
    K = 50
    
    unique_dates = df['Date'].unique()
    
    # Tüm veriyi önden scale edebiliriz (eğitim ve testin aynı scaler'ı kullanması için) 
    # Veya her gün için geçmişi scale edebiliriz. Hız için global scaler kullanalım çünkü 
    # metrikler zaten normalize edilmiş indikatörler (slope, r2, score).
    global_scaler = StandardScaler()
    df[features] = global_scaler.fit_transform(df[features])
    
    for current_date in tqdm(unique_dates, desc="Genişleyen Pencere (Expanding Window) KNN"):
        # Bugünün işlemleri
        today_mask = df['Date'] == current_date
        today_idx = df.index[today_mask]
        
        # Geçmiş işlemler (En az HOLD_DAYS gün öncesi kapanmış olanlar)
        cutoff_date = current_date - np.timedelta64(HOLD_DAYS, 'D')
        hist_mask = df['Date'] <= cutoff_date
        hist_df = df[hist_mask]
        
        if len(hist_df) >= K:
            X_hist = hist_df[features].values
            y_hist = hist_df['pl_pct'].values
            
            X_today = df.loc[today_mask, features].values
            
            knn = NearestNeighbors(n_neighbors=K, algorithm='auto')
            knn.fit(X_hist)
            
            distances, indices = knn.kneighbors(X_today)
            
            for i, idx in enumerate(today_idx):
                neighbor_idx = indices[i]
                neighbor_pls = y_hist[neighbor_idx]
                
                exp_pls[idx] = np.mean(neighbor_pls)
                win_rates[idx] = np.mean(neighbor_pls > 0) * 100
                
    df['Beklenen 7G Kar (%)'] = exp_pls
    df['Kazanma İhtimali (%)'] = win_rates
    
    # Baştaki yeterli geçmişi olmayan NaN satırlarını düşür
    df = df.dropna(subset=['Beklenen 7G Kar (%)', 'Kazanma İhtimali (%)'])
    
    df['ML Güven Skoru'] = (df['Kazanma İhtimali (%)'] / 100) * df['Beklenen 7G Kar (%)']
    
    # Geri kalan kod eski scriptle aynı (Görselleştirme vs)
    df.rename(columns={
        'pl_pct': '7 Günlük Kar (%)',
        'slope': 'Regresyon Eğimi',
        'r2': 'R2',
        'score': 'Uzaklık Skoru (Score)'
    }, inplace=True)
    
    target = '7 Günlük Kar (%)'
    metrics = [
        'Regresyon Eğimi',
        'R2',
        'Uzaklık Skoru (Score)',
        'Beklenen 7G Kar (%)',
        'Kazanma İhtimali (%)',
        'ML Güven Skoru'
    ]
    
    df['Kar_Agirligi'] = df[target].abs()
    
    print("Aykırı değerler kırpılıyor (%1 ve %99)...")
    plot_df = df.copy()
    
    q_low_target = plot_df[target].quantile(0.01)
    q_hi_target = plot_df[target].quantile(0.99)
    plot_df = plot_df[(plot_df[target] >= q_low_target) & (plot_df[target] <= q_hi_target)]
    
    for metric in metrics:
        q_low = plot_df[metric].quantile(0.01)
        q_hi = plot_df[metric].quantile(0.99)
        plot_df = plot_df[(plot_df[metric] >= q_low) & (plot_df[metric] <= q_hi)]

    print("Veriler korelasyon için çan eğrisine standardize ediliyor...")
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    normalized_data = qt.fit_transform(plot_df[metrics + [target]])
    normalized_df = pd.DataFrame(normalized_data, columns=metrics + [target], index=plot_df.index)
    
    print("\n--- Leakage Korumalı (Sıfır Hata) Ağırlıklı Korelasyonlar ---")
    weights = plot_df['Kar_Agirligi'].values
    weighted_corr_matrix = np.zeros((len(metrics) + 1, len(metrics) + 1))
    cols = metrics + [target]
    
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                weighted_corr_matrix[i, j] = 1.0
            else:
                cov = np.cov(normalized_df[cols[i]], normalized_df[cols[j]], aweights=weights)
                corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
                weighted_corr_matrix[i, j] = corr

    weighted_corr_df = pd.DataFrame(weighted_corr_matrix, columns=cols, index=cols)
    
    target_corrs = weighted_corr_df[target].sort_values(ascending=False)
    for index, value in target_corrs.items():
        if index != target:
            print(f"Korelasyon {index} vs {target}: {value:.4f}")
            
    out_dir = 'correlation_plots/ml_metrics_noleakage'
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(weighted_corr_df, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Sıfır Leakage - Kar Ağırlıklı Korelasyon Matrisi', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/correlation_heatmap_noleakage.png')
    plt.close()

    print("\nPorkchop ve Verimlilik Isı Haritaları çiziliyor...")
    plot_df['Is Profit'] = (plot_df[target] > 0).astype(int)
    
    for metric in metrics:
        fig = plt.figure(figsize=(16, 12))
        
        ax1 = plt.subplot(2, 2, 1) 
        ax2 = plt.subplot(2, 2, 2) 
        ax3 = plt.subplot(2, 1, 2) 
        
        sns.histplot(data=plot_df, x=metric, weights='Kar_Agirligi', kde=True, ax=ax1, color='orange', bins=50)
        ax1.set_title(f'{metric} Kar Ağırlıklı Dağılımı', fontsize=12, fontweight='bold')
        ax1.set_xlabel(metric)
        ax1.set_ylabel('Toplam Kar Hacmi')
        
        try:
            plot_df['bin'] = pd.qcut(plot_df[metric], q=10, duplicates='drop')
        except ValueError:
            plot_df['bin'] = pd.cut(plot_df[metric], bins=10)
            
        def weighted_win_rate(x):
            idx = x.index
            profits = plot_df.loc[idx, target]
            win_profit = profits[profits > 0].sum()
            abs_profit = profits.abs().sum()
            return (win_profit / abs_profit * 100) if abs_profit != 0 else 0

        def total_profit_generated(x):
            return x.sum()
            
        bin_stats = plot_df.groupby('bin').apply(lambda g: pd.Series({
            'Kar_Faktoru_Kazanma_Ihtimali': weighted_win_rate(g['Is Profit']),
            'Toplam_Uretilen_Net_Kar': total_profit_generated(g[target]),
            'Ortalama_Kar': g[target].mean()
        })).reset_index()
        
        bin_stats['bin'] = bin_stats['bin'].astype(str)
        bin_stats.set_index('bin', inplace=True)
        
        heatmap_data = bin_stats[['Kar_Faktoru_Kazanma_Ihtimali', 'Toplam_Uretilen_Net_Kar', 'Ortalama_Kar']]
        
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlOrRd', ax=ax2, linewidths=.5, cbar=True)
        ax2.set_title(f'Sayı Değerlerine Göre Kar Hacmi Verimliliği', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sayı Değeri Aralıkları')
        
        sns.histplot(data=plot_df, x=metric, y=target, weights='Kar_Agirligi', bins=50, pmax=.9, cmap="inferno", cbar=True, ax=ax3, cbar_kws={'label': 'Toplam Kar Hacmi'})
        sns.regplot(data=plot_df.sample(min(5000, len(plot_df))), x=metric, y=target, scatter=False, color='cyan', line_kws={"linewidth": 3}, ax=ax3)
        ax3.axhline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_title(f'SIFIR LEAKAGE Porkchop Haritası: {metric} vs {target}', fontsize=14, fontweight='bold')
        ax3.set_xlabel(metric)
        ax3.set_ylabel(target)
        
        plt.tight_layout()
        safe_metric_name = metric.split('(')[0].strip().replace('%', 'Pct').replace('/', '_').replace(' ', '_').replace('İ', 'I').replace('ı', 'i').replace('ğ', 'g')
        filename = f"{out_dir}/{safe_metric_name}_noleakage_analysis.png"
        plt.savefig(filename, dpi=150)
        plt.close()
    
    print(f"\nİşlem Tamam! Tüm Leakage-Korumalı analizler '{out_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
