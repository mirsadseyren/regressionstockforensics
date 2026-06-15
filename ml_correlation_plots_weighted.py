import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

def main():
    print("--- KAR AĞIRLIKLI (PROFIT-WEIGHTED) ML METRİKLERİ VE PORKCHOP ANALİZİ ---")
    
    # 1. Veri Yükleme
    print("Geçmiş işlem havuzu yükleniyor (historical_trade_metrics.csv)...")
    try:
        df = pd.read_csv('historical_trade_metrics.csv')
    except FileNotFoundError:
        print("Hata: historical_trade_metrics.csv bulunamadı.")
        return

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['slope', 'r2', 'score', 'pl_pct'])
    
    # 2. KNN ile ML Skorlarını Hesaplama
    features = ['slope', 'r2', 'score']
    X = df[features].values
    y = df['pl_pct'].values
    
    print("Özellikler ölçeklendiriliyor ve KNN çalıştırılıyor...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    K = 50
    knn = NearestNeighbors(n_neighbors=K+1, algorithm='auto', n_jobs=-1)
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(X_scaled)
    neighbor_indices = indices[:, 1:]
    neighbor_pls = y[neighbor_indices]
    
    exp_pls = np.mean(neighbor_pls, axis=1)
    win_rates = np.mean(neighbor_pls > 0, axis=1) * 100
    confidence_scores = (win_rates / 100) * exp_pls
    
    df['Beklenen 7G Kar (%)'] = exp_pls
    df['Kazanma İhtimali (%)'] = win_rates
    df['ML Güven Skoru'] = confidence_scores
    
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
    
    # Mutlak kar ağırlığı sütunu
    df['Kar_Agirligi'] = df[target].abs()
    
    # Aykırı değerleri kırpma (Görselleştirme ve Bining için)
    print("Aykırı değerler kırpılıyor (%1 ve %99)...")
    plot_df = df.copy()
    
    q_low_target = plot_df[target].quantile(0.01)
    q_hi_target = plot_df[target].quantile(0.99)
    plot_df = plot_df[(plot_df[target] >= q_low_target) & (plot_df[target] <= q_hi_target)]
    
    for metric in metrics:
        q_low = plot_df[metric].quantile(0.01)
        q_hi = plot_df[metric].quantile(0.99)
        plot_df = plot_df[(plot_df[metric] >= q_low) & (plot_df[metric] <= q_hi)]

    # --- STANDARDIZE VE ÇANLAŞTIRMA (Korelasyonlar için) ---
    print("Veriler korelasyon için çan eğrisine (Normal Dağılım) standardize ediliyor...")
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    normalized_data = qt.fit_transform(plot_df[metrics + [target]])
    normalized_df = pd.DataFrame(normalized_data, columns=metrics + [target], index=plot_df.index)
    
    # Ağırlıklı Korelasyon (Numpy cov ile)
    print("\n--- Kar Ağırlıklı (Weighted) Çanlaştırılmış Korelasyonlar ---")
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
            print(f"Ağırlıklı Korelasyon {index} vs {target}: {value:.4f}")
            
    # Klasör oluştur
    out_dir = 'correlation_plots/ml_metrics_weighted'
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(weighted_corr_df, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Kar Hacmi Ağırlıklı Korelasyon Matrisi (Çanlaştırılmış Veri)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/correlation_heatmap_weighted.png')
    plt.close()

    # --- PORKCHOP & VERİMLİLİK ISI HARİTALARI (Kar Ağırlıklı) ---
    print("\nKar Hacmine Göre Ağırlıklandırılmış Porkchop ve Isı Haritaları çiziliyor...")
    
    plot_df['Is Profit'] = (plot_df[target] > 0).astype(int)
    
    for metric in metrics:
        print(f"Ağırlıklı Grafik oluşturuluyor: {metric}...")
        
        fig = plt.figure(figsize=(16, 12))
        
        ax1 = plt.subplot(2, 2, 1) # Çan Eğrisi
        ax2 = plt.subplot(2, 2, 2) # Verimlilik Isı Haritası
        ax3 = plt.subplot(2, 1, 2) # Porkchop Plot
        
        # 1. Çan Eğrisi (Ağırlıklı Dağılım)
        sns.histplot(data=plot_df, x=metric, weights='Kar_Agirligi', kde=True, ax=ax1, color='orange', bins=50)
        ax1.set_title(f'{metric} Kar Ağırlıklı Dağılımı', fontsize=12, fontweight='bold')
        ax1.set_xlabel(metric)
        ax1.set_ylabel('Toplam Kar Hacmi (Mutlak)')
        
        # 2. Verimlilik Isı Haritası (Ağırlıklı Hesaplamalar)
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
        ax2.set_xlabel('Performans (Kar Hacmi)')
        
        # 3. Porkchop Yoğunluk Haritası (Ağırlıklı)
        sns.histplot(data=plot_df, x=metric, y=target, weights='Kar_Agirligi', bins=50, pmax=.9, cmap="inferno", cbar=True, ax=ax3, cbar_kws={'label': 'Toplam Kar Hacmi (Mutlak)'})
        
        # Ortalamayı gösteren trend çizgisi
        sns.regplot(data=plot_df.sample(min(5000, len(plot_df))), x=metric, y=target, scatter=False, color='cyan', line_kws={"linewidth": 3}, ax=ax3)
        
        ax3.axhline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
        
        ax3.set_title(f'AĞIRLIKLI Porkchop Haritası (Kar Hacmi): {metric} vs {target}', fontsize=14, fontweight='bold')
        ax3.set_xlabel(f"{metric} (Orijinal Sayı Değerleri)", fontsize=12)
        ax3.set_ylabel(target, fontsize=12)
        
        plt.tight_layout()
        safe_metric_name = metric.split('(')[0].strip().replace('%', 'Pct').replace('/', '_').replace(' ', '_').replace('İ', 'I').replace('ı', 'i').replace('ğ', 'g')
        filename = f"{out_dir}/{safe_metric_name}_weighted_analysis.png"
        plt.savefig(filename, dpi=150)
        plt.close()
    
    print(f"\nİşlem Tamam! Kar hacmine göre ağırlıklandırılmış tüm grafikler '{out_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
